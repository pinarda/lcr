#!/bin/tcsh

set arrname = ( `seq 0 116` )
foreach x ($arrname)
  cat CNN_script.sh | sed "s/-vn 0/-vn $x/" > scripts/$x.sh
end