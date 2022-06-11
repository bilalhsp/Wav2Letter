from wav2letter.utils.levenshtein import levenshtein_distance

def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):

    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer