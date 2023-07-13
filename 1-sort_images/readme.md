# Hierarchical Organization

This script helps to reorganize patches of different resolutions.

Given the initial folder path (e.g. x5, x10, x20) it will reorganize the patches into


# Example:
```
Slide_id
tumor_001_tumor/
    0_x_21056_y_119296:
        0__x_21056_y_119296/
            0__x_21568_y_119808.jpg
        0__x_21056_y_119296.jpg
        0__x_21056_y_120320/
            0__x_21056_y_120320.jpg
            0__x_21056_y_120832.jpg
            0__x_21568_y_120320.jpg
            0__x_21568_y_120832.jpg
        0__x_21056_y_120320.jpg
        0__x_22080_y_119296/
            0__x_22080_y_119296.jpg
            0__x_22080_y_119808.jpg
            0__x_22592_y_119296.jpg
            0__x_22592_y_119808.jpg
        0__x_22080_y_119296.jpg
        0__x_22080_y_120320/
            0__x_22080_y_120320.jpg
            0__x_22080_y_120832.jpg
            0__x_22592_y_120320.jpg
            0__x_22592_y_120832.jpg
        0__x_22080_y_120320.jpg
    0_x_21056_y_119296.jpg
    /0_x_21056_y_121344
        0__x_21056_y_121344/
            0__x_21056_y_121344.jpg
            0__x_21056_y_121856.jpg
            0__x_21568_y_121344.jpg
            0__x_21568_y_121856.jpg
        0__x_21056_y_121344.jpg
        0__x_22080_y_121344/
            0__x_22080_y_121344.jpg
            0__x_22080_y_121856.jpg
            0__x_22592_y_121344.jpg
            0__x_22592_y_121856.jpg
        0__x_22080_y_121344.jpg
    0_x_21056_y_121344.jpg
```


# Launch

```
python sort_hierarchy --sourcex5  x5PATH --sourcex10 x10PATH --sourcex20 x20PATH --dest DEST
```
