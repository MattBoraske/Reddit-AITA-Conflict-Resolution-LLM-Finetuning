import zstandard
import os
import json
import sys
import csv
from datetime import datetime
import logging.handlers

########################
# FILTERING PARAMETERS #
########################
##  input file
#input_file = r"base-dumps\base-dumps\AmItheAsshole_comments.zst"

##  output file. the file extension from below will be added automatically
#output_file = r"comments_2019_to_2022_at_least_10_score"

##  output file extension: 3 options
##      zst: same as the input, a zstandard compressed ndjson file. Can be read by the other scripts in the repo
##      txt: an ndjson file, which is a text file with a separate json object on each line. Can be opened by any text editor
##      csv: a comma separated value file. Can be opened by a text editor or excel
#output_format = "zst"

##  option to override the above format and output only a single field into a text file, one per line. Use to make lists of IDs
##     id: the id of the submission or comment
##     link_id: only for comments, the fullname of the submission the comment is associated with
##     parent_id: only for comments, the fullname of the parent of the comment. Either another comment or the submission if it's top level
#single_field = None

##  option to write out to the log every time there's a bad line, set to false if expecting only some lines to match the key
write_bad_lines = False

##  from/to date filters
date_filtering = False
from_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
to_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

##  min/max score filters
score_filtering = False
min_score = 10
max_score = 1000000 # impossibly high to negate limit

## BASE DRIVER CONFIG ##
#field = None # field to filter on
#values = [''] # values to filter the filed on
#values_file = None # file input if values list is long (ex. a .txt containing a list of submission IDs)
#exact_match = False # match if similar (field contains the value) or exact (field is exactly the value)

### STEPS TO GET SUBMISSIONS AND RESPECTIVE TOP-LEVEL COMMENTS OUTPUT FILE ###
## 1. Get a filtered submissions .zst
#
## 2. Use the filtered submissions .zst as input for another run of the script we'll run the script again with the same input and filters, but set the output to single field.
#input_file = "submissions_2019_to_2022_above_50_score.zst"
#output_file = "submissions_2019_to_2022_above_50_score_ids"
#single_field = "id"
#
# 3. Remove all the other filters and update the script to input from the comments file, and use the submission ids list we created before as the values list filter.
input_file = "new_datasets\comments_2019_to_2022_at_least_10_score.zst"
output_file = "new_datasets\\top_level_comments_2019_to_2022_at_least_10_comment_score_at_least_50_submission_score"
output_format = "zst"
single_field = None  # resetting this back so it's not used
field = "parent_id"  # in the comment object, this is the field that contains the submission id. if you want only top level comments instead of all comments, you can set this to "parent_id" instead of "link_id"
values_file = "new_datasets\submissions_2019_to_2022_at_least_50_score_ids.txt"
values = [""]
exact_match = False  # the link_id field has a prefix on it, so we can't do an exact match

# logging configuration
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)
if not os.path.exists("logs"):
	os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024*1024*16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)


def write_line_zst(handle, line):
	"""
    Writes a line to a file handle in zstd-compressed format.

    Parameters:
    - handle: The file handle to write to, expected to be a zstd-compressed stream.
    - line: The line (string) to be written.
    """
	handle.write(line.encode('utf-8'))
	handle.write("\n".encode('utf-8'))


def write_line_json(handle, obj):
    """
    Writes a JSON object to a file handle as a JSON-formatted string.

    Parameters:
    - handle: The file handle to write to.
    - obj: The JSON object to be written.
    """
    handle.write(json.dumps(obj))
    handle.write("\n")


def write_line_single(handle, obj, field):
	"""
    Writes a single field of a JSON object to a file handle.

    Parameters:
    - handle: The file handle to write to.
    - obj: The JSON object containing the field.
    - field: The specific field from the JSON object to write.
    """
	if field in obj:
		handle.write(obj[field])
	else:
		log.info(f"{field} not in object {obj['id']}")
	handle.write("\n")


def write_line_csv(writer, obj, is_submission):
	"""
    Writes a line to a CSV file from a JSON object.

    Parameters:
    - writer: A CSV writer object.
    - obj: The JSON object to extract data from.
    - is_submission: A boolean indicating whether the object is a Reddit submission.
    """
	output_list = []
	output_list.append(str(obj['score']))
	output_list.append(datetime.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d"))
	if is_submission:
		output_list.append(obj['title'])
	output_list.append(f"u/{obj['author']}")
	output_list.append(f"https://www.reddit.com{obj['permalink']}")
	if is_submission:
		if obj['is_self']:
			if 'selftext' in obj:
				output_list.append(obj['selftext'])
			else:
				output_list.append("")
		else:
			output_list.append(obj['url'])
	else:
		output_list.append(obj['body'])
	writer.writerow(output_list)


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	"""
    Reads and decodes a chunk of data from a reader object, handling UnicodeDecodeError.

    Parameters:
    - reader: The reader object to read from.
    - chunk_size: Size of the chunk to read in bytes.
    - max_window_size: Maximum window size for reading to avoid infinite loops.
    - previous_chunk: The previously read chunk of data (for concatenation).
    - bytes_read: Total number of bytes read so far.

    Returns:
    - Decoded chunk of data as a string.
    """
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
	"""
    Generator function to read and yield lines from a zstd-compressed file.

    Parameters:
    - file_name: The name of the file to be read.

    Yields:
    - Tuple of the line read and the current file handle position.
    """
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")
			for line in lines[:-1]:
				yield line.strip(), file_handle.tell()
			buffer = lines[-1]
		reader.close()


def process_file(input_file, output_file, output_format, field, values, from_date, to_date, single_field, exact_match):
	"""
    Processes an input file and writes the filtered content to an output file in a specified format.

    Parameters:
    - input_file: Path of the file to process.
    - output_file: Base path for the output file.
    - output_format: Format of the output file (zst, txt, csv).
    - field: The field used for filtering the data.
    - values: List of values to match in the specified field.
    - from_date, to_date: Date range for filtering the data.
    - single_field: A specific field to write in case of txt format.
    - exact_match: Boolean indicating whether to match values exactly or partially.

    Function Logic:
    Opens the input file, applies various filters (date, score, field matching), and writes the filtered data to the output file in the specified format.
    """
	output_path = f"{output_file}.{output_format}"
	is_submission = "submission" in input_file
	log.info(f"Input: {input_file} : Output: {output_path} : Is submission {is_submission}")
	writer = None
	if output_format == "zst":
		handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))
	elif output_format == "txt":
		handle = open(output_path, 'w', encoding='UTF-8')
	elif output_format == "csv":
		handle = open(output_path, 'w', encoding='UTF-8', newline='')
		writer = csv.writer(handle)
	else:
		log.error(f"Unsupported output format {output_format}")
		sys.exit()

	file_size = os.stat(input_file).st_size
	created = None
	matched_lines = 0
	bad_lines = 0
	total_lines = 0
	for line, file_bytes_processed in read_lines_zst(input_file):
		total_lines += 1
		if total_lines % 100000 == 0:
			log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : {matched_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

		try:
			obj = json.loads(line)
			created = datetime.utcfromtimestamp(int(obj['created_utc']))
			if date_filtering:
				if created < from_date:
					continue
				if created > to_date:
					continue
			
			score = int(obj['score'])
			if score_filtering:
				score = int(obj['score'])
				if score < min_score:
					continue
				if score > max_score:
					continue
                                   						           
			if field is not None:
				field_value = obj[field].lower()
				matched = False
				for value in values:
					if exact_match:
						if value == field_value:
							matched = True
							break
					else:
						if value in field_value:
							matched = True
							break
				if not matched:
					continue

			matched_lines += 1
			if output_format == "zst":
				write_line_zst(handle, line)
			elif output_format == "csv":
				write_line_csv(writer, obj, is_submission)
			elif output_format == "txt":
				if single_field is not None:
					write_line_single(handle, obj, single_field)
				else:
					write_line_json(handle, obj)
			else:
				log.info(f"Something went wrong, invalid output format {output_format}")
		except (KeyError, json.JSONDecodeError) as err:
			bad_lines += 1
			if write_bad_lines:
				if isinstance(err, KeyError):
					log.warning(f"Key {field} is not in the object: {err}")
				elif isinstance(err, json.JSONDecodeError):
					log.warning(f"Line decoding failed: {err}")
				log.warning(line)

	handle.close()
	log.info(f"Complete : {total_lines:,} : {matched_lines:,} : {bad_lines:,}")


if __name__ == "__main__":
    if single_field is not None:
        # if single_field is specified, switch output format to plain text
        log.info("Single field output mode, changing output file format to txt")
        output_format = "txt"

    if values_file is not None:
        # if a values file is provided, load values from it
        values = []
        with open(values_file, 'r') as values_handle:
            for value in values_handle:
                values.append(value.strip().lower())  # strip whitespace and convert to lowercase
        log.info(f"Loaded {len(values)} from values file {values_file}")
    else:
        # if no values file is provided, use the provided values list and convert to lowercase
        values = [value.lower() for value in values]

    # logging information about the filters and settings being used
    log.info(f"Filtering field: {field}")
    '''if len(values) <= 20:
        log.info(f"On values: {','.join(values)}")
    else:
        # if there are many values, log them individually
        log.info(f"On values:")
        for value in values:
            log.info(value)'''
    log.info(f"Exact match {('on' if exact_match else 'off')}. Single field {single_field}.")
    log.info(f"From date {from_date.strftime('%Y-%m-%d')} to date {to_date.strftime('%Y-%m-%d')}")
    log.info(f"Output format set to {output_format}")

    # determining input files to process
    input_files = []
    if os.path.isdir(input_file):
        # if input_file is a directory, process all .zst files in it
        if not os.path.exists(output_file):
            os.makedirs(output_file)  # Create output directory if it doesn't exist
        for file in os.listdir(input_file):
            if not os.path.isdir(file) and file.endswith(".zst"):
                # build input and output file paths for each file
                input_name = os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]
                input_files.append((os.path.join(input_file, file), os.path.join(output_file, input_name)))
    else:
        # if input_file is a single file, add it directly to the processing list
        input_files.append((input_file, output_file))

    # logging the number of files to be processed and starting the processing
    log.info(f"Processing {len(input_files)} files")
    for file_in, file_out in input_files:
        # for each file pair, call process_file with the specified parameters
        process_file(file_in, file_out, output_format, field, values, from_date, to_date, single_field, exact_match)
