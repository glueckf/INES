#Author Mr P. Meran
#https://github.com/PMeran101/INEv/blob/main/INEv/scripts/sim_PP_Query.sh
#!/bin/sh
# modified by Mr N. Dilek on 23.07.2025

cd ../code
	
  echo "Script started at: $(date)"


  for k in 5 10 15 20 25 # qwl size
  do 
      for h in 6 10 15 20 25 # num event types
      do
        for j in 4 5 6 7 8 9 10
        do

      
          echo "Starting outer loop with k=$k and h=$h at: $(date)"


     
          a=1
          while [ "$a" -lt 200 ]
          do
              echo "  Starting inner loop iteration $a with k=$k at: $(date)"
            python3 start_simulation.py -nw 50 -ner 0.5 -ne "$h" -es 1.3 -mp 4 --count $k --length $j
            
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