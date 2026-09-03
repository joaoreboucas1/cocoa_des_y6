export OUTPUT_PIP_1="/dev/null"
export OUTPUT_PIP_2="/dev/null"

python plot1.py > ${OUTPUT_PIP_1} 2> ${OUTPUT_PIP_2} 

python plot2.py > ${OUTPUT_PIP_1} 2> ${OUTPUT_PIP_2}

python plot3.py > ${OUTPUT_PIP_1} 2> ${OUTPUT_PIP_2}