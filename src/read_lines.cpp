#include "read_lines_extension.hpp"
#include "line_selection.hpp"
#include "compat.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/types/data_chunk.hpp"

namespace duckdb {

// =============================================================================
// Memory-bounded streaming line reader for NON-SEEKABLE sources.
//
// A pipe / virtual URI (shellfs, scalarfs) cannot be rewound and has no
// reliable file size, so EOF must be detected by a 0-byte Read() rather than by
// comparing SeekPosition() against GetFileSize() (the latter throws on a pipe,
// which previously caused a BLANK LINE mid-stream to be misread as EOF and
// truncate the source). This reader NEVER buffers the whole stream: it keeps a
// fixed refill buffer plus, at most, one in-progress (partial) line carried
// across refills. Resident memory is therefore bounded by REFILL_SIZE + the
// length of the single longest line.
//
// Line endings handled: "\n", "\r\n", and bare "\r". A final line without a
// trailing newline is emitted. Blank lines mid-stream are emitted as empty
// content. line_number and byte_offset are tracked arithmetically;
// Seek/SeekPosition/GetFileSize are never called on the handle.
// =============================================================================
struct StreamLineReader {
	static constexpr idx_t REFILL_SIZE = 65536;

	FileHandle *file = nullptr;
	string carry;        // bytes read past the last complete line (in-progress line)
	idx_t carry_pos;     // scan/consume cursor within `carry`
	int64_t byte_offset; // absolute byte offset of the next line's first byte
	bool eof;            // underlying Read() has returned 0
	bool done;           // EOF reached AND carry fully drained

	StreamLineReader() : carry_pos(0), byte_offset(0), eof(false), done(true) {
	}

	void Reset(FileHandle &handle) {
		file = &handle;
		carry.clear();
		carry_pos = 0;
		byte_offset = 0;
		eof = false;
		done = false;
	}

	// Pull one more REFILL_SIZE block from the stream into `carry`, compacting
	// the already-consumed prefix first so resident memory stays bounded.
	// Returns false when the underlying stream is exhausted (0-byte read).
	bool Refill() {
		if (eof) {
			return false;
		}
		// Drop the consumed prefix so `carry` only holds the live partial line.
		if (carry_pos > 0) {
			carry.erase(0, carry_pos);
			carry_pos = 0;
		}
		char buffer[REFILL_SIZE];
		int64_t bytes_read = file->Read(buffer, REFILL_SIZE);
		if (bytes_read <= 0) {
			eof = true;
			return false;
		}
		carry.append(buffer, static_cast<size_t>(bytes_read));
		return true;
	}

	// Try to split one complete line out of `carry` starting at carry_pos.
	// On success sets `line` (content including its terminator) and advances
	// carry_pos. Returns false if no complete line is currently buffered (the
	// caller must Refill, or flush the remainder at EOF).
	bool TrySplitLine(string &line) {
		idx_t i = carry_pos;
		const idx_t n = carry.size();
		while (i < n) {
			char c = carry[i];
			if (c == '\n') {
				idx_t end = i + 1;
				line = carry.substr(carry_pos, end - carry_pos);
				carry_pos = end;
				return true;
			}
			if (c == '\r') {
				idx_t end = i + 1;
				if (end < n) {
					if (carry[end] == '\n') {
						end++; // "\r\n"
					}
					line = carry.substr(carry_pos, end - carry_pos);
					carry_pos = end;
					return true;
				}
				// A trailing '\r' at the very end of the buffer is ambiguous: it
				// could be a bare-'\r' terminator or the first half of "\r\n".
				// If the stream is exhausted it is a terminator; otherwise defer
				// until the next refill disambiguates.
				if (eof) {
					line = carry.substr(carry_pos, end - carry_pos);
					carry_pos = end;
					return true;
				}
				return false;
			}
			i++;
		}
		return false;
	}

	// Read the next logical line. Returns false once the stream is fully
	// consumed. On true, `line` includes its terminator (if any) and `offset` is
	// the absolute byte offset of the line's first byte.
	bool NextLine(string &line, int64_t &offset) {
		if (done) {
			return false;
		}
		while (true) {
			if (TrySplitLine(line)) {
				offset = byte_offset;
				byte_offset += static_cast<int64_t>(line.size());
				return true;
			}
			if (Refill()) {
				continue;
			}
			// Stream exhausted: flush any remaining bytes as the final line (a
			// last line without a trailing newline, or a trailing bare '\r').
			if (carry_pos < carry.size()) {
				line = carry.substr(carry_pos);
				offset = byte_offset;
				byte_offset += static_cast<int64_t>(line.size());
				carry_pos = carry.size();
				done = true;
				return true;
			}
			done = true;
			return false;
		}
	}
};

struct ReadTextLinesBindData : public TableFunctionData {
	vector<OpenFileInfo> files;
	LineSelection line_selection;
	bool ignore_errors;

	ReadTextLinesBindData(vector<OpenFileInfo> files, LineSelection selection, bool ignore_errors)
	    : files(std::move(files)), line_selection(std::move(selection)), ignore_errors(ignore_errors) {
	}
};

struct ReadTextLinesGlobalState : public GlobalTableFunctionState {
	idx_t file_index;
	unique_ptr<FileHandle> current_file;
	int64_t current_line_number;
	int64_t current_byte_offset;
	string current_file_path;
	bool file_finished;
	FileSystem *fs;
	LineSelection resolved_selection; // Per-file resolved selection (handles from-end refs)

	// Non-seekable (pipe / stream) support: when the current source cannot seek,
	// lines are produced by a memory-bounded streaming reader instead of
	// FileHandle::ReadLine + Seek-based EOF probing. The reader state persists in
	// global state across the chunked invocations of this scan.
	bool current_seekable;
	StreamLineReader stream;

	ReadTextLinesGlobalState()
	    : file_index(0), current_line_number(0), current_byte_offset(0), file_finished(true), fs(nullptr),
	      resolved_selection(LineSelection::All()), current_seekable(true) {
	}

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> ReadTextLinesBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto input_path = input.inputs[0].GetValue<string>();

	// Try the original path first - if it exists or matches files, use it as-is
	// This handles cases where filenames contain colons (e.g., "file:2.txt")
	auto files = compat::GlobFilesCompat(fs, input_path, context, FileGlobOptions::ALLOW_EMPTY);

	string glob_pattern = input_path;
	LineSelection path_line_selection = LineSelection::All();

	if (files.empty()) {
		// No files found with original path - try parsing for embedded line spec
		auto parsed_result = LineSelection::ParsePathWithLineSpec(input_path);
		if (parsed_result.first != input_path) {
			// Path was parsed differently, try globbing with the extracted path
			files = compat::GlobFilesCompat(fs, parsed_result.first, context, FileGlobOptions::ALLOW_EMPTY);
			if (!files.empty()) {
				glob_pattern = parsed_result.first;
				path_line_selection = std::move(parsed_result.second);
			}
		}
	}

	LineSelection line_selection = LineSelection::All();
	bool has_explicit_lines = false;
	int64_t before_context = 0;
	int64_t after_context = 0;
	bool ignore_errors = false;

	// Check for second positional argument (lines)
	if (input.inputs.size() > 1 && !input.inputs[1].IsNull()) {
		line_selection = LineSelection::Parse(input.inputs[1]);
		has_explicit_lines = true;
	}

	for (auto &param : input.named_parameters) {
		auto &name = param.first;
		auto &value = param.second;

		if (name == "lines") {
			line_selection = LineSelection::Parse(value);
			has_explicit_lines = true;
		} else if (name == "before") {
			before_context = value.GetValue<int64_t>();
		} else if (name == "after") {
			after_context = value.GetValue<int64_t>();
		} else if (name == "context") {
			before_context = value.GetValue<int64_t>();
			after_context = before_context;
		} else if (name == "ignore_errors") {
			ignore_errors = value.GetValue<bool>();
		}
	}

	// If no explicit lines param, use path-embedded selection
	if (!has_explicit_lines && !path_line_selection.IsAll()) {
		line_selection = std::move(path_line_selection);
	}

	if (before_context > 0 || after_context > 0) {
		line_selection.AddContext(before_context, after_context);
	}

	return_types.push_back(LogicalType::BIGINT);
	names.push_back("line_number");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("content");

	return_types.push_back(LogicalType::BIGINT);
	names.push_back("byte_offset");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("file_path");

	if (files.empty() && !ignore_errors) {
		throw IOException("No files found that match the pattern \"%s\"", input_path);
	}

	return make_uniq<ReadTextLinesBindData>(std::move(files), std::move(line_selection), ignore_errors);
}

static unique_ptr<GlobalTableFunctionState> ReadTextLinesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<ReadTextLinesGlobalState>();
	result->fs = &FileSystem::GetFileSystem(context);
	return std::move(result);
}

// Count total lines in a file (for resolving from-end references)
static int64_t CountLinesInFile(FileHandle &file) {
	int64_t count = 0;

	try {
		(void)file.SeekPosition();
		file.Seek(0);
	} catch (const std::exception &e) {
		throw IOException("from-end line selection requires a seekable source. "
		                  "Pipes and streams do not support Seek/SeekPosition. "
		                  "Use only positive line numbers/ranges. Details: %s",
		                  e.what());
	}

	while (true) {
		try {
			string line = file.ReadLine();
			if (line.empty()) {
				auto current_pos = file.SeekPosition();
				auto file_size = file.GetFileSize();
				if (current_pos >= file_size) {
					break;
				}
			}
			count++;
		} catch (...) {
			break;
		}
	}
	file.Seek(0);
	return count;
}

static bool OpenNextFile(ReadTextLinesGlobalState &state, const ReadTextLinesBindData &bind_data) {
	while (state.file_index < bind_data.files.size()) {
		auto &file_info = bind_data.files[state.file_index];
		state.file_index++;

		try {
			state.current_file = state.fs->OpenFile(file_info.path, FileFlags::FILE_FLAGS_READ);
			state.current_file_path = file_info.path;
			state.current_line_number = 0;
			state.current_byte_offset = 0;
			state.file_finished = false;
			state.current_seekable = state.current_file->CanSeek();

			if (!state.current_seekable) {
				// Non-seekable source (pipe / virtual URI): stream lines with the
				// memory-bounded reader. from-end references are impossible on a
				// pure stream without buffering everything, so reject them here
				// (forward/positive selections work).
				if (bind_data.line_selection.HasFromEndReferences()) {
					throw IOException("from-end line selection requires a seekable source. "
					                  "Pipes and streams do not support Seek/SeekPosition. "
					                  "Use only positive line numbers/ranges.");
				}
				state.resolved_selection = bind_data.line_selection;
				state.stream.Reset(*state.current_file);
				return true;
			}

			// Handle from-end references (e.g., +10 meaning 10th line from end)
			if (bind_data.line_selection.HasFromEndReferences()) {
				// Count lines first, then resolve
				int64_t total_lines = CountLinesInFile(*state.current_file);
				state.resolved_selection = bind_data.line_selection;
				state.resolved_selection.ResolveFromEnd(total_lines);
			} else {
				state.resolved_selection = bind_data.line_selection;
			}

			return true;
		} catch (std::exception &e) {
			if (!bind_data.ignore_errors) {
				throw;
			}
			continue;
		}
	}
	return false;
}

static void ReadTextLinesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ReadTextLinesBindData>();
	auto &state = data_p.global_state->Cast<ReadTextLinesGlobalState>();

	idx_t output_row = 0;

	while (output_row < STANDARD_VECTOR_SIZE) {
		if (state.file_finished) {
			if (!OpenNextFile(state, bind_data)) {
				break;
			}
		}

		// Non-seekable sources are served by the memory-bounded streaming reader:
		// EOF is a 0-byte read, blank lines mid-stream are preserved, and a final
		// line without a trailing newline is emitted. Reader state persists across
		// chunks via global state.
		if (!state.current_seekable) {
			while (output_row < STANDARD_VECTOR_SIZE && !state.file_finished) {
				string line;
				int64_t line_start_offset = 0;
				if (!state.stream.NextLine(line, line_start_offset)) {
					state.file_finished = true;
					break;
				}

				state.current_line_number++;

				if (!state.resolved_selection.ShouldIncludeLine(state.current_line_number)) {
					if (state.resolved_selection.PastAllRanges(state.current_line_number)) {
						state.file_finished = true;
						break;
					}
					continue;
				}

				output.data[0].SetValue(output_row, Value::BIGINT(state.current_line_number));
				output.data[1].SetValue(output_row, Value(line));
				output.data[2].SetValue(output_row, Value::BIGINT(line_start_offset));
				output.data[3].SetValue(output_row, Value(state.current_file_path));

				output_row++;
			}
			continue;
		}

		while (output_row < STANDARD_VECTOR_SIZE && !state.file_finished) {
			string line;
			auto line_start_offset = state.current_byte_offset;

			try {
				line = state.current_file->ReadLine();
			} catch (...) {
				state.file_finished = true;
				break;
			}

			if (line.empty()) {
				bool at_eof = true;
				try {
					auto current_pos = state.current_file->SeekPosition();
					auto file_size = state.current_file->GetFileSize();
					if (current_pos < file_size) {
						at_eof = false;
					}
				} catch (const std::exception &) {
					at_eof = true;
				}
				if (at_eof) {
					state.file_finished = true;
					break;
				}
			}

			state.current_line_number++;
			try {
				state.current_byte_offset = state.current_file->SeekPosition();
			} catch (const std::exception &) {
				state.current_byte_offset = line_start_offset + static_cast<int64_t>(line.size()) + 1;
			}

			if (!state.resolved_selection.ShouldIncludeLine(state.current_line_number)) {
				if (state.resolved_selection.PastAllRanges(state.current_line_number)) {
					state.file_finished = true;
					break;
				}
				continue;
			}

			output.data[0].SetValue(output_row, Value::BIGINT(state.current_line_number));
			output.data[1].SetValue(output_row, Value(line));
			output.data[2].SetValue(output_row, Value::BIGINT(line_start_offset));
			output.data[3].SetValue(output_row, Value(state.current_file_path));

			output_row++;
		}
	}

	output.SetCardinality(output_row);
}

TableFunctionSet ReadLinesFunction() {
	TableFunctionSet set("read_lines");

	// Single argument: read_lines(path)
	TableFunction func1("read_lines", {LogicalType::VARCHAR}, ReadTextLinesFunction, ReadTextLinesBind,
	                    ReadTextLinesInit);
	func1.named_parameters["lines"] = LogicalType::ANY;
	func1.named_parameters["before"] = LogicalType::BIGINT;
	func1.named_parameters["after"] = LogicalType::BIGINT;
	func1.named_parameters["context"] = LogicalType::BIGINT;
	func1.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	set.AddFunction(func1);

	// Two arguments: read_lines(path, lines)
	TableFunction func2("read_lines", {LogicalType::VARCHAR, LogicalType::ANY}, ReadTextLinesFunction,
	                    ReadTextLinesBind, ReadTextLinesInit);
	func2.named_parameters["before"] = LogicalType::BIGINT;
	func2.named_parameters["after"] = LogicalType::BIGINT;
	func2.named_parameters["context"] = LogicalType::BIGINT;
	func2.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	set.AddFunction(func2);

	return set;
}

// =============================================================================
// Lateral join version: read_lines_lateral
// =============================================================================

struct ReadTextLinesLateralBindData : public TableFunctionData {
	LineSelection line_selection;
	bool ignore_errors;

	ReadTextLinesLateralBindData(LineSelection selection, bool ignore_errors)
	    : line_selection(std::move(selection)), ignore_errors(ignore_errors) {
	}
};

struct ReadTextLinesLateralState : public LocalTableFunctionState {
	FileSystem *fs;
	unique_ptr<FileHandle> current_file;
	string current_file_path;
	int64_t current_line_number;
	int64_t current_byte_offset;
	bool file_open;
	idx_t current_row;
	LineSelection resolved_selection; // Per-file resolved selection
	idx_t chunks_processed;

	// Non-seekable (pipe / stream) support. The InOut operator is re-entered
	// (HAVE_MORE_OUTPUT) until the current source is exhausted, so the streaming
	// reader state (refill buffer, carry, offset, eof/done) must persist in
	// operator state across re-invocations. Each correlated input row runs its
	// own command and gets its own reader.
	bool current_seekable;
	StreamLineReader stream;

	ReadTextLinesLateralState()
	    : fs(nullptr), current_line_number(0), current_byte_offset(0), file_open(false), current_row(0),
	      resolved_selection(LineSelection::All()), chunks_processed(0), current_seekable(true) {
	}
};

static unique_ptr<FunctionData> ReadTextLinesLateralBind(ClientContext &context, TableFunctionBindInput &input,
                                                         vector<LogicalType> &return_types, vector<string> &names) {
	LineSelection line_selection = LineSelection::All();
	bool ignore_errors = false;

	// For in_out functions, additional positional arguments appear in input_table_names.
	// The second argument (lines selection) is at index 1.
	if (input.input_table_names.size() > 1) {
		// The argument value is stored as the "column name"
		string lines_arg = input.input_table_names[1];

		// Strip surrounding quotes if present (string literals come with quotes)
		if (lines_arg.size() >= 2 && lines_arg.front() == '\'' && lines_arg.back() == '\'') {
			lines_arg = lines_arg.substr(1, lines_arg.size() - 2);
		}

		if (!lines_arg.empty()) {
			// Parse as string - LineSelection::Parse handles both integers and line specs
			line_selection = LineSelection::Parse(Value(lines_arg));
		}
	}

	// Check for second positional argument (lines)
	// Note: Named parameters don't work with in_out functions, so we only support positional.
	// Context can be embedded in the lines spec (e.g., '42 +/-3').
	if (input.inputs.size() > 1 && !input.inputs[1].IsNull()) {
		line_selection = LineSelection::Parse(input.inputs[1]);
	}

	return_types.push_back(LogicalType::BIGINT);
	names.push_back("line_number");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("content");

	return_types.push_back(LogicalType::BIGINT);
	names.push_back("byte_offset");

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("file_path");

	return make_uniq<ReadTextLinesLateralBindData>(std::move(line_selection), ignore_errors);
}

static unique_ptr<LocalTableFunctionState> ReadTextLinesLateralLocalInit(ExecutionContext &context,
                                                                         TableFunctionInitInput &input,
                                                                         GlobalTableFunctionState *global_state) {
	auto result = make_uniq<ReadTextLinesLateralState>();
	result->fs = &FileSystem::GetFileSystem(context.client);
	return std::move(result);
}

static OperatorResultType ReadTextLinesLateralInOut(ExecutionContext &context, TableFunctionInput &data_p,
                                                    DataChunk &input, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ReadTextLinesLateralBindData>();
	auto &state = data_p.local_state->Cast<ReadTextLinesLateralState>();

	if (input.size() == 0) {
		output.SetCardinality(0);
		return OperatorResultType::FINISHED;
	}

	idx_t output_row = 0;

	while (output_row < STANDARD_VECTOR_SIZE) {
		// Need to open a new file?
		if (!state.file_open) {
			if (state.current_row >= input.size()) {
				// All input rows for this chunk are processed. If we produced
				// output rows in THIS invocation we must flush them first (the
				// pipeline contract forbids returning a terminal/NEED_MORE_INPUT
				// signal together with a non-empty output chunk); the terminal
				// decision is then taken on the next, empty re-invocation.
				if (output_row > 0) {
					output.SetCardinality(output_row);
					return OperatorResultType::HAVE_MORE_OUTPUT;
				}
				output.SetCardinality(0);
				state.current_row = 0;
				state.chunks_processed++;
				if (input.size() == 1 && state.chunks_processed >= 1) {
					return OperatorResultType::FINISHED;
				}
				return OperatorResultType::NEED_MORE_INPUT;
			}

			// Get file path from input
			auto path_value = input.GetValue(0, state.current_row);
			if (path_value.IsNull()) {
				state.current_row++;
				continue;
			}

			auto file_path = path_value.GetValue<string>();

			// Try to open the file
			try {
				state.current_file = state.fs->OpenFile(file_path, FileFlags::FILE_FLAGS_READ);
				state.current_file_path = file_path;
				state.current_line_number = 0;
				state.current_byte_offset = 0;
				state.file_open = true;

				// Probe seekability via CanSeek() (side-effect-free; true for
				// local files, false for pipes/streams) rather than perturbing
				// the read position with SeekPosition().
				state.current_seekable = state.current_file->CanSeek();

				if (!state.current_seekable) {
					// Non-seekable source (e.g. a per-row shellfs pipe in a
					// correlated lateral join): stream lines with the
					// memory-bounded reader. The reader state lives in operator
					// state so it survives the HAVE_MORE_OUTPUT re-invocations of
					// this operator. from-end references on a pure stream are
					// impossible without buffering everything, so reject them.
					if (bind_data.line_selection.HasFromEndReferences()) {
						throw IOException("from-end line selection requires a seekable source. "
						                  "Pipes and streams do not support Seek/SeekPosition. "
						                  "Use only positive line numbers/ranges.");
					}
					state.resolved_selection = bind_data.line_selection;
					state.stream.Reset(*state.current_file);
				} else if (bind_data.line_selection.HasFromEndReferences()) {
					// Handle from-end references (e.g., +10 meaning 10th line from end)
					int64_t total_lines = CountLinesInFile(*state.current_file);
					state.resolved_selection = bind_data.line_selection;
					state.resolved_selection.ResolveFromEnd(total_lines);
				} else {
					state.resolved_selection = bind_data.line_selection;
				}
			} catch (std::exception &e) {
				if (!bind_data.ignore_errors) {
					throw;
				}
				state.current_row++;
				continue;
			}
		}

		// Read lines from the current source. Non-seekable sources are served by
		// the memory-bounded streaming reader so blank lines mid-stream and a
		// final line without a trailing newline survive, and so the reader
		// position persists across this operator's HAVE_MORE_OUTPUT re-invocations.
		if (!state.current_seekable) {
			while (output_row < STANDARD_VECTOR_SIZE && state.file_open) {
				string line;
				int64_t line_start_offset = 0;
				if (!state.stream.NextLine(line, line_start_offset)) {
					state.file_open = false;
					state.current_row++;
					break;
				}

				state.current_line_number++;

				// Check line selection
				if (!state.resolved_selection.ShouldIncludeLine(state.current_line_number)) {
					if (state.resolved_selection.PastAllRanges(state.current_line_number)) {
						state.file_open = false;
						state.current_row++;
						break;
					}
					continue;
				}

				// Output the line
				output.data[0].SetValue(output_row, Value::BIGINT(state.current_line_number));
				output.data[1].SetValue(output_row, Value(line));
				output.data[2].SetValue(output_row, Value::BIGINT(line_start_offset));
				output.data[3].SetValue(output_row, Value(state.current_file_path));

				output_row++;
			}
		} else {
			while (output_row < STANDARD_VECTOR_SIZE && state.file_open) {
				string line;
				auto line_start_offset = state.current_byte_offset;

				try {
					line = state.current_file->ReadLine();
				} catch (...) {
					state.file_open = false;
					state.current_row++;
					break;
				}

				// Check for EOF
				if (line.empty()) {
					bool at_eof = true;
					try {
						auto current_pos = state.current_file->SeekPosition();
						auto file_size = state.current_file->GetFileSize();
						if (current_pos < file_size) {
							at_eof = false;
						}
					} catch (const std::exception &) {
						at_eof = true;
					}
					if (at_eof) {
						state.file_open = false;
						state.current_row++;
						break;
					}
				}

				state.current_line_number++;
				try {
					state.current_byte_offset = state.current_file->SeekPosition();
				} catch (const std::exception &) {
					state.current_byte_offset = line_start_offset + static_cast<int64_t>(line.size()) + 1;
				}

				// Check line selection
				if (!state.resolved_selection.ShouldIncludeLine(state.current_line_number)) {
					if (state.resolved_selection.PastAllRanges(state.current_line_number)) {
						state.file_open = false;
						state.current_row++;
						break;
					}
					continue;
				}

				// Output the line
				output.data[0].SetValue(output_row, Value::BIGINT(state.current_line_number));
				output.data[1].SetValue(output_row, Value(line));
				output.data[2].SetValue(output_row, Value::BIGINT(line_start_offset));
				output.data[3].SetValue(output_row, Value(state.current_file_path));

				output_row++;
			}
		}

		// If file closed and more rows, continue to next file
		if (!state.file_open && state.current_row < input.size()) {
			continue;
		}

		// Exit if we have output or no more work
		if (output_row > 0 || state.current_row >= input.size()) {
			break;
		}
	}

	output.SetCardinality(output_row);

	// More output from current file?
	if (state.file_open) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}

	// More input rows to process?
	if (state.current_row < input.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}

	// The current source(s) drained in this invocation. If we emitted rows we
	// must flush them with HAVE_MORE_OUTPUT and defer the terminal signal to the
	// next (empty) call: the pipeline asserts that FINISHED / NEED_MORE_INPUT is
	// only returned with an empty output chunk. This matters for streaming
	// (non-seekable) sources, which can fully drain within a single invocation.
	if (output_row > 0) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}

	state.current_row = 0;
	state.chunks_processed++;
	if (input.size() == 1 && state.chunks_processed >= 1) {
		return OperatorResultType::FINISHED;
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

TableFunctionSet ReadLinesLateralFunction() {
	TableFunctionSet set("read_lines_lateral");

	// Single argument: read_lines_lateral(path)
	TableFunction func1("read_lines_lateral", {LogicalType::VARCHAR}, nullptr, ReadTextLinesLateralBind, nullptr,
	                    ReadTextLinesLateralLocalInit);
	func1.in_out_function = ReadTextLinesLateralInOut;
	set.AddFunction(func1);

	// Two arguments: read_lines_lateral(path, lines)
	// Note: Named parameters don't work with in_out functions, so we only support positional.
	// Context can be embedded in the lines spec (e.g., '+5 +/-2').
	TableFunction func2("read_lines_lateral", {LogicalType::VARCHAR, LogicalType::ANY}, nullptr,
	                    ReadTextLinesLateralBind, nullptr, ReadTextLinesLateralLocalInit);
	func2.in_out_function = ReadTextLinesLateralInOut;
	set.AddFunction(func2);

	return set;
}

} // namespace duckdb
