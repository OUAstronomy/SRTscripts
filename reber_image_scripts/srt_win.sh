#!/usr/bin/bash

# screen shot directory
scrot_dir='/home/jjtobin/public_html/window_images'

# read all pids with srt in name
ps ax | grep ./srtn | awk '{print $1}' > /tmp/del1
total=$(ps ax | grep ./srtn | awk '{print $1}' | head -n 2)
p=$(wc -l "/tmp/del1" | awk '{print $1}')

# counter to ensure only when window is open that images are made
counter=0

# now for sorting
if [ ${p} -gt 1 ]; then
    findpid=$(head -n 1 /tmp/del1 | awk '{print $1}')
    known_windows=$(xwininfo -display :1 -root -tree | sed -e 's/^ *//' | grep -E "^0x" | awk '{print $1}')

    if [ ! -f /home/jjtobin/public_html/.srt_win_count ]; then
        echo "0" > /home/jjtobin/public_html/.srt_win_count
    fi

    count=$(cat /home/jjtobin/public_html/.srt_win_count)
    for id in ${known_windows}
    do
      pid=$(xprop -display :1 -id $id WM_NAME | cut -d'=' -f2 | tr -d ' ' | tr -d '"')
      if [ "${pid}" = "srtn" ]
      then
        temp=$(xwininfo -display :1 -id $id | grep 'Map State' | awk '{print $3}')
        if [ $temp = 'IsViewable' ]; then
          srtwin=$id
          counter=1
        fi
      fi
    done

    if [ $count -ge 43200 ]; then
        countc=$[ $count%43200 ]
    else
        countc=$count
    fi

    if [ $counter -eq 1 ]; then
        import -display :1 -window "$srtwin" "$scrot_dir/srt_prog_$countc.jpg"
        echo $(($count+1)) > "/home/jjtobin/public_html/.srt_win_count"
    fi
fi

