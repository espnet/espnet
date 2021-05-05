id=SP18_131
break_flag=0
for broken_id in SP18_119 SP18_122 SP18_124 SP18_125 SP18_132 SP18_133 SP18_141 SP18_189 SP18_228 SP16_168 SP15_32 SP07_154; do
  if [ $id == $broken_id ]; then
    echo "Remove $id from dataset."
    break_flag=1
  fi
done

if [ $break_flag -eq 1 ];then
  echo "broken"        
else
  echo "$id"
fi
