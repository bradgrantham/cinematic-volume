if [ a$3 == a ]
then
    echo usage: $0 first count dicom-root-dir/
    exit 1
fi

FIRST=$1
COUNT=$2
LAST=`expr $FIRST + $COUNT - 1`
YOUR_DICOM_DATA=$3

for i in `seq $FIRST $LAST`
do
    index=`expr $i - 12`
    FILE=file_`printf %03d $index`.bin
    echo extracting binary data $FILE from DICOM file $i...
    python3 ./extract-one.py $YOUR_DICOM_DATA/DICOM/$i $FILE
done
