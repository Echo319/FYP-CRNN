# By merge
# CHAR_VECTOR = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"
# By class
UPPER_CASE = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
ALL_LETTERS = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
LOWER_CASE = u'abcdefghijklmnopqrstuvwxyz'
LOWER_CASE_SPACE = u'abcdefghijklmnopqrstuvwxyz '

letters = [letter for letter in LOWER_CASE]

num_classes = len(letters) + 1

img_w, img_h = 128, 64

# Network parameters
batch_size = 128
val_batch_size = 64

downsample_factor = 4
max_text_len = 9