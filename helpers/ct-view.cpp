#include <iostream>

#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

DcmDirectoryRecord* imageRecord = NULL;
DcmStack imageStack;

void processSeries(DcmDirectoryRecord* seriesRecord)
{
    std::vector<DcmDirectoryRecord*> imageRecords;

    // ??

    printf("%zd\n", imageRecords.size());
}

int main(int argc, const char *argv[])
{
    if(argc < 2) {
        printf("usage: %s dicomdir [series]\n", argv[0]);
        printf("    Providing no series name lists the SERIES records in the file.\n");
        exit(EXIT_FAILURE);
    }

    std::string dicom_name{argv[1]};
    std::string series_name;
    if(argc > 2) {
        series_name = argv[2];
    }

    printf("loading %s\n", dicom_name.c_str());
    DcmDicomDir dir(dicom_name.c_str());  // Load the DICOMDIR file.
    OFCondition status = dir.error();

    if (status.bad()) {
        std::cerr << "Problem opening file:" << dicom_name << std::endl;
        exit(EXIT_FAILURE);
    }

    dir.print(std::cout);

    // DcmDataset* dataset = file_format.getDataset();
    // if(dataset == nullptr) {
        // printf("couldn't load dataset\n");
        // exit(EXIT_FAILURE);
    // }

    // Get the root record (typically this is a "PATIENT" record).
    DcmDirectoryRecord* rootRecord = &(dir.getRootRecord());

    bool complete = false;

    if(series_name.empty()) {
        printf("series:\n");
        complete = true;
    }

    DcmDirectoryRecord* patientRecord = nullptr;
    while ((patientRecord = rootRecord->nextSub(patientRecord)) != nullptr) {
        DcmDirectoryRecord* studyRecord = nullptr;
        while ((studyRecord = patientRecord->nextSub(studyRecord)) != nullptr) {
            DcmDirectoryRecord* seriesRecord = nullptr;
            while ((seriesRecord = studyRecord->nextSub(seriesRecord)) != nullptr) {
                OFString seriesDescription;
                OFCondition result = seriesRecord->findAndGetOFString(DCM_SeriesDescription, seriesDescription);
                if(series_name.empty()) {
                    printf("\t%s\n", seriesDescription.c_str());
                } else {
                    if (result.good() && seriesDescription.compare(series_name) == 0) {
                        std::cout << "Found " << series_name << "\n";
                        processSeries(seriesRecord);
                        // Additional code to work with the series.
                        complete = true;
                    }
                }
            }
        }
    }

    if(!complete) {
        printf("Didn't find %s\n", series_name.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


