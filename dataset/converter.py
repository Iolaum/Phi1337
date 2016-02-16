import csv
import json

csvfile = open('test.csv', 'r')
jsonfile = open('test.json', 'w')

fieldnames = ("id", "product_uid", "product_title", "search_term")

reader = csv.DictReader(csvfile, fieldnames)

for row in reader:
    json.dumps(row, jsonfile)
    jsonfile.write('\n')
