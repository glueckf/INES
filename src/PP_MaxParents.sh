#!/bin/sh

	
  echo "Script started at: $(date)"

  for k in 5 10 20 # qwl size
  do 
      for h in 6 10 15 20 25 # num event types
      do

      
          echo "Starting outer loop with k=$k and h=$h at: $(date)"


     
          a=1
          while [ "$a" -lt 200 ]
          do
              echo "  Starting inner loop iteration $a with k=$k at: $(date)"
            python3 start_simulation.py -nw 50 -ner 0.5 -ne "$h" -es 1.3 -mp 1 --count $k --length 6           
            echo "  Completed inner loop iteration $a with k=$k at: $(date)"
              a=$((a + 1))
              sleep 1
        #   done
      done    
          echo "Completed outer loop with k=$k and h=$h at: $(date)"
      done
      done
  done
  
  echo "Script completed at: $(date)"