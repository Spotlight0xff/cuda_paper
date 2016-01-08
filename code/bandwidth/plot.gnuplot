reset
set border 31 linewidth .3 # thin border
set xlabel "Size [Bytes]"
set format y '%.1fGB'
set ytics 1,2,15
set yrange [1:15]
set ylabel "Bandwidth [GB/s]"
#set logscale x
set key left top
set terminal png size 400,300 enhanced font "Helvetica,20"
set output 'bandwidth_pageable_vs_pinned_overall.png'
#plot "data128_pageable" u 1:2 with linespoints title "Pageable Memory (H2D)" ,\
     #"data128_pageable" u 1:3 with linespoints title "Pageable Memory (D2H)" ,\
     #"data128_pinned" u 1:2 with linespoints title "Pinned Memory (H2D)" ,\
     #"data128_pinned" u 1:3 with linespoints title "Pinned Memory (D2H)" 

plot "data128_pageable" u 1:4 with linespoints title "Pageable Memory" ,\
     "data128_pinned" u 1:4 with points title "Pinned Memory" 
#plot "Banglapedia.dat" using 1:($2/1e6) with linespoints title 'Banglapedia' ,\
#     "World_Factbook.dat" using 1:($2/1e6) with linespoints title 'World Factbook'

#set output
#set terminal qt persist
reset
