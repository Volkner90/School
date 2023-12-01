import xml.etree.ElementTree as ET
import csv
import os

# Initialize an empty list to store the extracted data
data = []

# Iterate over all XML files in your folder
xml_folder = r"C:\Users\jonathan\data" #Your file directory here
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, xml_file)
        
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract information from each "object" element
        for obj in root.findall(".//object"):
            name = obj.find("name").text
            xmin = obj.find(".//xmin").text
            ymin = obj.find(".//ymin").text
            xmax = obj.find(".//xmax").text
            ymax = obj.find(".//ymax").text

            # Get the path from the "path" element
            path = root.find(".//path").text

            # Append the extracted data to the list
            data.append([path, xmin, ymin, xmax, ymax, name])

# Write the extracted data to a CSV file
csv_file = "labels.csv"
with open(csv_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Path", "xmin", "ymin", "xmax", "ymax", "Name"])
    csv_writer.writerows(data)
