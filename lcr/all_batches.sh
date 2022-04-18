set arr=(bc_a1_SRF)
foreach x ($arr)
qsub cheyenne_batch_$x.sh
end