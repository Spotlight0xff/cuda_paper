samples <- 10
num <- 128
data_pageable = array(0,c(num,4))
data_pinned = array(0, c(num,4))
for(count in seq(1:samples)) {
    file_pageable = paste("data",num,"_pageable",count, sep="")
    pageable = read.table(file_pageable)
    file_pinned = paste("data",num,"_pinned",count, sep="")
    pinned = read.table(file_pinned)
    for (i in seq(1:num)) {
        data_pageable[i,1] = pageable[i,1] # size
        data_pageable[i,2] = data_pageable[i,2] + pageable[i,2] # h2d
        data_pageable[i,3] = data_pageable[i,3] + pageable[i,3] # d2h
        data_pageable[i,4] = data_pageable[i,4] + pageable[i,4] # both
        data_pinned[i,1] = pinned[i,1] # size
        data_pinned[i,2] = data_pinned[i,2] + pinned[i,2] # h2d
        data_pinned[i,3] = data_pinned[i,3] + pinned[i,3] # d2h
        data_pinned[i,4] = data_pinned[i,4] + pinned[i,4] # both
    }
}

for( i in seq(1:num)) {
    data_pinned[i,2] = data_pinned[i,2] / samples;
    data_pinned[i,3] = data_pinned[i,3] / samples;
    data_pinned[i,4] = data_pinned[i,4] / samples;
    data_pageable[i,2] = data_pageable[i,2] / samples;
    data_pageable[i,3] = data_pageable[i,3] / samples;
    data_pageable[i,4] = data_pageable[i,4] / samples;
}
file_pageable = paste("data",num,"_pageable", sep="")
file_pinned = paste("data", num, "_pinned", sep="")
write.table(data_pageable, file=file_pageable, quote=F, append=T, sep="\t", eol="\n",dec=".", 
             row.names=F, col.names=F)
write.table(data_pinned, file=file_pinned, quote=F, append=T, sep="\t", eol="\n",dec=".", 
             row.names=F, col.names=F)
#print(data_pinned)
