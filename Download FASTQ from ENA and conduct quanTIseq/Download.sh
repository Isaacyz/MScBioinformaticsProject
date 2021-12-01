Download_url=./urls/
output=./output/
if [ ! -d "$output" ]; then
        mkdir $output
fi

for FILE in $Download_url*
do	
	wget -i $FILE
	 
	prefix="$(basename $FILE)"
	prefix=$(echo $prefix | cut -d'.' -f -1)
	quanTIseqTest_input=$prefix'_quantiseq_input_file.txt'
	# Need to have quanTIseq pipeline
	bash quanTIseq_pipeline.sh --inputfile=$quanTIseqTest_input --outputdir=./output/ --prefix=$prefix --rawcounts=TRUE

	rm *.fastq.gz
done

