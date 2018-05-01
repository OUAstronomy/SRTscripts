#!/usr/bin/bash

# directory
python_dir_orig='/home/jjtobin/RealTime-Sun'
python_dir_fin='/home/jjtobin/public_html/python_integrated'
count_dir='/home/jjtobin/public_html/.realsun_int_count'

# pull file 
if [ ! -f ${count_dir} ]; then
    echo "0" > ${count_dir}
fi
count=$(cat ${count_dir})

if [ $count -ge 43200 ]; then
    countc=$[ $count%43200 ]
else
    countc=$count
fi

cp -f "${python_dir_orig}/sun-current.png" "${python_dir_fin}/sun_current_${countc}.png"
cp -f "${python_dir_orig}/sun-current.all.png" "/home/jjtobin/public_html/img/sun-current.all.png"
cp -f "${python_dir_orig}/sun-current.all.lowres.png" "/home/jjtobin/public_html/img/sun-current.all.lowres.png"
echo $(($count+1)) > ${count_dir}
