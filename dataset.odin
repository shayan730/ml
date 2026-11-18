package main

import "core:encoding/csv"
import "core:fmt"
import "core:os"
import "core:strconv"

Dataset :: struct {
	x:             [dynamic]f64,
	y:             [dynamic]f64, //TODO consider Enum?
	class_ids:     [dynamic]int,
	feature_count: int,
	sample_count:  int,
}

dataset_load :: proc(ds: ^Dataset, filename: string, allocator := context.allocator) {
	reader: csv.Reader
	reader.trim_leading_space = true
	reader.reuse_record = true
	reader.reuse_record_buffer = true
	defer csv.reader_destroy(&reader)

	csv_data, csv_err := os.read_entire_file(filename, allocator)
	defer delete(csv_data)

	if csv_err == nil {
		csv.reader_init_with_string(&reader, string(csv_data), allocator)
	} else {
		fmt.eprintfln("Unable to open file: %v. Error: %v", filename, csv_err)
	}

	for row, i, row_err in csv.iterator_next(&reader) {
		if row_err != nil {fmt.eprintfln("Error reading row: %v. Error: %v", row_err, row_err)}
		if i == 0 {
			ds.feature_count = len(row) - 1
		} else {
			ds.sample_count += 1
			for col, col_idx in row {
				if col_idx == 0 {
					append(&ds.y, strconv.parse_f64(col) or_else -1.0)
				} else {
					append(&ds.x, strconv.parse_f64(col) or_else -1.0)
				}
			}
		}
	}
	assert(len(ds.x) == ds.sample_count * ds.feature_count)
	fmt.printf(
		"Successfully loaded %v. feature count: %v - sample count: %v\n",
		filename,
		ds.feature_count,
		ds.sample_count,
	)
}

dataset_destroy :: proc(ds: ^Dataset) {
	delete(ds.x)
	delete(ds.y)
}
