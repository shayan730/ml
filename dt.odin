package main

import "core:fmt"

Node :: struct {
	label:       u64,
	feature_idx: int,
	threshold:   f64,
	is_leaf:     bool,
	left:        ^Node,
	right:       ^Node,
}

Split :: struct {
	feature_idx: int,
	threshold:   f64,
	gini:        f64,
	is_valid:    bool,
}

find_best_split :: proc(ds: ^Dataset, rows: []int) -> Split {
	if len(rows) <= 1 {
		return Split{}
	}

	best_split := Split {
		gini = 1.0,
	}

	for feature_idx in 0 ..< ds.feature_count {
		// Each observed feature value is tried as a threshold.
		for row in rows {
			threshold := ds.x[row * ds.feature_count + feature_idx]
			left_counts := make(map[int]int, 0, context.temp_allocator)
			right_counts := make(map[int]int, 0, context.temp_allocator)
			left_n := 0
			right_n := 0

			for other_row in rows {
				label := int(ds.y[other_row])
				value := ds.x[other_row * ds.feature_count + feature_idx]

				if value <= threshold {
					left_counts[label] += 1
					left_n += 1
				} else {
					right_counts[label] += 1
					right_n += 1
				}
			}

			if left_n == 0 || right_n == 0 {
				continue
			}

			left_gini := 1.0
			for _, count in left_counts {
				p := f64(count) / f64(left_n)
				left_gini -= p * p
			}

			right_gini := 1.0
			for _, count in right_counts {
				p := f64(count) / f64(right_n)
				right_gini -= p * p
			}

			n := f64(left_n + right_n)
			weighted_gini := (f64(left_n) / n) * left_gini + (f64(right_n) / n) * right_gini

			if weighted_gini < best_split.gini {
				best_split = Split {
					feature_idx = feature_idx,
					threshold   = threshold,
					gini        = weighted_gini,
					is_valid    = true,
				}
			}
		}
	}

	return best_split
}

build_tree :: proc(ds: ^Dataset, rows: []int, depth, max_depth, min_samples_split: int) -> ^Node {
	node := new(Node, context.allocator)

	label_counts := make(map[int]int, 0, context.temp_allocator)
	for row in rows {
		label_counts[int(ds.y[row])] += 1
	}

	best_count := 0
	for label, count in label_counts {
		if count > best_count {
			node.label = u64(label)
			best_count = count
		}
	}

	if len(rows) == 0 {
		node.is_leaf = true
		return node
	}

	pure := true
	first_label := ds.y[rows[0]]
	for row in rows {
		if ds.y[row] != first_label {
			pure = false
			break
		}
	}

	if pure || depth >= max_depth || len(rows) < min_samples_split {
		node.is_leaf = true
		return node
	}

	split := find_best_split(ds, rows)
	if !split.is_valid {
		node.is_leaf = true
		return node
	}

	left_rows := make([dynamic]int, 0, len(rows), context.temp_allocator)
	right_rows := make([dynamic]int, 0, len(rows), context.temp_allocator)

	for row in rows {
		value := ds.x[row * ds.feature_count + split.feature_idx]
		if value <= split.threshold {
			append(&left_rows, row)
		} else {
			append(&right_rows, row)
		}
	}

	if len(left_rows) == 0 || len(right_rows) == 0 {
		delete(left_rows)
		delete(right_rows)
		node.is_leaf = true
		return node
	}

	node.is_leaf = false
	node.feature_idx = split.feature_idx
	node.threshold = split.threshold
	node.left = build_tree(ds, left_rows[:], depth + 1, max_depth, min_samples_split)
	node.right = build_tree(ds, right_rows[:], depth + 1, max_depth, min_samples_split)

	delete(left_rows)
	delete(right_rows)
	return node
}


tree_destroy :: proc(n: ^Node) {
	if n == nil {
		return
	}
	tree_destroy(n.left)
	tree_destroy(n.right)
	free(n)
}

count_nodes :: proc(n: ^Node) -> int {
	if n == nil {
		return 0
	}
	return 1 + count_nodes(n.left) + count_nodes(n.right)
}

predict :: proc(ds: ^Dataset, n: ^Node, sample_row: int) -> u64 {
	if n.is_leaf {
		return n.label
	}
	value := ds.x[sample_row * ds.feature_count + n.feature_idx]
	if value <= n.threshold {
		return predict(ds, n.left, sample_row)
	} else {
		return predict(ds, n.right, sample_row)
	}
}
