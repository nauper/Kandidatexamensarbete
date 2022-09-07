# a class for summing up the csv files

import csv

csvFile = "C:\\Users\\BIGGER MAC\\Desktop\\New folder (3)\\csv/combined.csv"
csvReader = csv.DictReader(open(csvFile, newline=''))

with open(csvFile+"v", 'w', newline='') as csvfile:
    # fieldnames = ["Subject ID","Sex","Weight","Research Group","Age","Modality","Description","Imaging Protocol","Image ID","Plane","Weighting","Slice Thickness"]


    fieldnames = ["Image ID", "Has Parkinson", "Weight", "Sex", "Age", "Subject ID"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in csvReader:
        hasParkinson = 0  # 0 = control, 1  = parkinson
        sex = 0 # 0 = female, 1 = male
        plane = 0 # CORONAL = 0 ,          1 = AXIAL, 2 = SAG
        weighting = 0 # PD = 0,  T1 = 1,  T2 = 2
        #print(row)
        if(row['Research Group'] == 'PD'):
            hasParkinson = 1
        if(row['Sex'] == 'M'):
            sex = 1

        # if(row['Plane'] == 'AXIAL'):
        #     plane = 1
        # if(row['Plane'] == 'SAGITTAL'):
        #     plane = 2

        # if (row['Weighting'] == 'T1'):
        #     weighting = 1
        # if(row['Weighting'] == 'T2'):
        #     weighting = 2
        writer.writerow({"Image ID": row['Image ID'],"Has Parkinson": hasParkinson, "Weight": row['Weight'], "Sex": sex, "Age": row['Age'], "Subject ID": row['Subject ID']})

        #writer.writerow({"Image ID": row['Image ID'],"Has Parkinson": hasParkinson, "Weight": row['Weight'], "Sex": sex, "Age": row['Age'],"Plane": plane, "Weighting": weighting, "Subject ID": row['Subject ID']})
        print("Image ID", row['Image ID'],"Has Parkinson", hasParkinson, "Weight", row['Weight'], "Sex", sex, "Age", row['Age'],"Plane", plane, "Weighting", weighting,"Subject ID", row['Subject ID'])
