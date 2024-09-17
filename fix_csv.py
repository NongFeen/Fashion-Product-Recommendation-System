import random

def replace_comma_in_first_10_fields(file_input, file_output):
    flag = True
    with open(file_input, 'r', encoding='utf-8') as infile, open(file_output, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.split(',')
            first_10_parts = ';'.join(parts[:9])
            remaining_parts = ','.join(parts[9:])
            price = 'price'
            if(flag):
                flag = False
            else:
                price = str(int(random.uniform(1, 10))*pow(10,int(random.uniform(1,4))))
            
            new_line = first_10_parts + ";" + price + ";" + remaining_parts 
            outfile.write(new_line  + '')

# เรียกใช้ฟังก์ชัน โดยกำหนดไฟล์ input และ output
replace_comma_in_first_10_fields('dataset/styles.csv', 'dataset/fix_styles.csv')