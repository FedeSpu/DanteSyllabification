from src.utils.roman_to_int import *


# Pre-processing text for better handling
def preprocessing_text(file_name, file_write = False):
    punctuation = '•:?!<>.;,-—«»“”()'
    text_raw = open('../text/' + file_name + '.txt', 'rb').readlines()
    verses = []

    for line in text_raw:
        line_dec = line.decode(encoding='utf-8').lstrip()

        if not line_dec:
            # Append "end of triplet", useful for special rhetorical forms. Only one, even if more new lines
            if verses and (verses[-1] != ("<EOT>" or "<EOC>")):
                verses.append("<EOT>")
            continue
        if '•' in line_dec:
            # Get number of "Canto"
            line_split = line_dec.split('Canto', 1)
            book = line_split[0].strip()
            canticle_roman = line_split[1].strip()
            canticle = roman_to_decimal(canticle_roman)
            '''
            # Add "end of canticle"
            if canticle > 1:
                verses.append("<EOC>")
            '''
            # verses.append(book + " Canto " + str(canticle))
            continue

        for p in punctuation:
            if p in line_dec:
                line_dec = line_dec.replace(p, '')

        verses.append(line_dec.rstrip())

    if file_write:
        f2 = open('../outputs/' + file_name + '_verses.txt', 'wb')
        for x in verses:
            f2.write((x + '\n').encode('utf-8'))

        f2.close()
    return verses
