package main

import "core:math"

validate_pair :: proc(y_true: []f64, y_pred: []f64) -> bool {
	return len(y_true) > 0 && len(y_true) == len(y_pred)
}

rmse :: proc(y_true: []f64, y_pred: []f64) -> (f64, bool) {
	if !validate_pair(y_true, y_pred) {
		return 0, false
	}
	squared_sum := 0.0
	for i in 0 ..< len(y_true) {
		d := y_true[i] - y_pred[i]
		squared_sum += d * d
	}

	return math.sqrt_f64(squared_sum / f64(len(y_true))), true // y_true's size being recalculated.
}

mae :: proc(y_true: []f64, y_pred: []f64) -> (f64, bool) {
	if !validate_pair(y_true, y_pred) {
		return 0, false
	}

	absolute_sum := 0.0
	for i in 0 ..< len(y_true) {
		d := y_true[i] - y_pred[i] // repeated operation
		absolute_sum += math.abs(d)
	}

	return absolute_sum / f64(len(y_true)), true // repeated op
}
