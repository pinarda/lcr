#!/bin/tcsh

set arrname=(15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360 375 390 405 420 435 450 465 480 495 510 525 540 555 570 585 600 615 630 645 660 675 690 705 720)

./all_batches_3d.sh
foreach x ($arrname)
@ y = $x + 15
./all_batches_3d_$x\_$y.sh
end
./all_batches.sh
./all_batches_modified.sh
./all_batches_modified_2.sh
./all_batches_modified_3.sh