package main

import "core:fmt"
import "core:mem"
import "core:os"

main :: proc() {
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v \n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v \n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}


	train_set: Dataset
	test_set: Dataset

	dataset_load(&train_set, os.args[1])
	dataset_load(&test_set, os.args[2])
	defer dataset_destroy(&train_set)
	defer dataset_destroy(&test_set)

	rows := make([dynamic]int, 0, len(train_set.y), context.allocator)
	for i in 0 ..< len(train_set.y) {
		append(&rows, i)
	}

	root := build_tree(&train_set, rows[:], 0, 5, 2)
	delete(rows)

	node_count := count_nodes(root)
	fmt.println("Number of trained nodes:", node_count)

	correct_pred := 0
	for sample_row in 0..<30 {
		y_true := test_set.y[sample_row]
		y_pred := predict(&test_set, root, sample_row)
		fmt.printfln("y_true: %v y_pred: %v", y_true, y_pred)
		if y_true == f64(y_pred) {correct_pred += 1}
	}
	fmt.printfln("Accuracy %.2f%%", f64(correct_pred) / f64(test_set.sample_count))

	tree_destroy(root)
	fmt.printf("\nSuccess! Success!\n")

}
