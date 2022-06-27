#!/bin/tcsh

set arrname=(15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 360 375 390 405 420 435 450 465 480 495 510 525 540 555 570 585 600 615 630 645 660 675 690 705 720)
set varname=(CCN3 CLOUD FLNS FLNT FSNS FSNT LHFLX PRECC PRECL PS QFLX SHFLX TMQ TS U)

foreach x ($arrname)
@ y = $x + 15
mkdir batch\_scripts/3d\_dssim\_scripts\_$x\_$y
foreach z ($varname)
cat batch\_scripts/3d\_dssim\_scripts/cheyenne\_batch_$z.sh | sed "s/-ts 0 -tt 15/-ts $x -tt $y/" > batch\_scripts/3d\_dssim\_scripts\_$x\_$y/cheyenne\_batch\_$z.sh
cat batch\_scripts/3d\_dssim\_scripts/$z.json > batch\_scripts/3d\_dssim\_scripts\_$x\_$y/$z.json
end
cat all_batches_3d.sh | sed "s/3d_dssim_scripts/3d_dssim_scripts\_$x\_$y/" > all_batches_3d_$x\_$y.sh
end
