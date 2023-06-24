def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def contains_only_letters(string):
    for char in string:
        if not char.isalpha():
            return False
    return True

def hard_rules_GR(text, Evidence_type):
    # Hard-rule for 'CCV:00012'
    if 'Greek' == text: return 'CCV:00012'
    # Hard-rule for 'CCV:00013'
    if text in ['Male', 'Female']: return 'CCV:00013'
    # Hard-rule for 'CCV:00014'
    if is_float(text) and Evidence_type == 'Passport': return 'CCV:00014'    
    # Hard-rule for 'CCV:00022'
    if len(text) == 9 and contains_only_letters(text[:2]) and text[2] == ' ' and text[3:].isdigit(): return 'CCV:00022'
    # Hard-rule for 'CCV:00026'
    if 'ΑΤ,' in text and Evidence_type in ['CriminalReIDcord', 'ID']: return 'CCV:00026'
    # Hard-rule for 'CCV:00028'
    if 'Ληξιαρχείο' in text: return 'CCV:00028'
    # Hard-rule for 'CCV:00029'
    if 'ΚΕΠ' in text and Evidence_type == 'BirthCertificate': return 'CCV:00029'
    # Hard-rule for 'CCV:00034'
    if len(text) == 9 and contains_only_letters(text[:2]) and text[2:].isdigit(): return 'CCV:00034'
    # Hard-rule for 'CCV:00038'
    if 'GRE' == text: return 'CCV:00038'
    # Hard-rule for 'CCV:00041'
    if len(text) == 11 and text.isdigit(): return 'CCV:00041'
    # Hard-rule for 'CCV:00053'
    if Evidence_type == 'PrimarySchool' and text.isdigit() and len(text) == 8: return 'CCV:00053'
    # Hard-rule for 'CCV:00054'
    if Evidence_type == 'PrimarySchool' and text.isdigit() and len(text) <= 2: return 'CCV:00054'
    # Hard-rule for 'CCV:00057'
    if 'Γυμνάσιο' in text: return 'CCV:00057'
    # Hard-rule for 'CCV:00058'
    if Evidence_type == 'LowerSecondarySchool' and text.isdigit() and len(text) == 8: return 'CCV:00058'
    # Hard-rule for 'CCV:00059'
    if Evidence_type == 'LowerSecondarySchool' and is_float(text) and len(text) < 8: return 'CCV:00059'
    # Hard-rule for 'CCV:00061'
    if 'Λύκειο' in text: return 'CCV:00061'
    # Hard-rule for 'CCV:00062'
    if Evidence_type == 'HigherSecondarySchool' and text.isdigit() and len(text) == 8: return 'CCV:00062'
    # Hard-rule for 'CCV:00063'
    if Evidence_type == 'HigherSecondarySchool' and is_float(text) and len(text) < 8: return 'CCV:00063'
    # Hard-rule for 'CCV:00066'
    if Evidence_type == 'TertiarySchool' and text.isdigit() and len(text) == 8: return 'CCV:00066'
    # Hard-rule for 'CCV:00067'
    if Evidence_type == 'TertiarySchool' and text.isdigit() and len(text) <= 2: return 'CCV:00067'
    # Hard-rule for 'CCV:00068'
    if Evidence_type == 'TertiarySchool' and text in ['University', 'MedicalSchool', 'TechnicalSchool', 'PolytechnicSchool', 'TradeSchool']: return 'CCV:00068'
    # Hard-rule for 'CCV:00069'
    if Evidence_type == 'TertiarySchool' and text.isdigit() and len(text) == 3: return 'CCV:00069'
    # Hard-rule for 'CCV:00071'    
    if '%' in text and is_float(text[:-1]): return 'CCV:00071'
    # Hard-rule for 'CCV:00073'
    if 'Νοσοκομείο' in text and Evidence_type == 'DisabilityRecord' : return 'CCV:00073' 
    # Hard-rule for 'CCV:00080'
    if 'ΑΤ,' in text and Evidence_type == 'CriminalRecord': return 'CCV:00080'
    # Hard-rule for 'CCV:00081'
    if text in ['True', 'False']: return 'CCV:00081' 
    # Hard-rule for 'CCV:00083'
    if text in ['Protected', 'Insured', 'Uninsured']: return 'CCV:00083'
    # Hard-rule for 'CCV:00085'
    if '[' in text and Evidence_type == 'DisabilityRecord': return 'CCV:00085'
    # Hard-rule for 'CCV:00089'
    if 'Νοσοκομείο' in text and Evidence_type == 'MedicalRecord' : return 'CCV:00089'
    # Hard-rule for 'CCV:00090'
    if '[' in text and Evidence_type == 'MedicalRecord': return 'CCV:00090'
    # Hard-rule for 'CCV:00094'
    if 'ΚΕΠ' in text and Evidence_type == 'ResidenceCertificate': return 'CCV:00094'
    # Hard-rule for 'CCV:000197'
    if 'bank' in text or 'Bank' in text: return 'CCV:00097'
    # Hard-rule for 'CCV:00098'
    if 'GR' in text and len(text) == 24: return 'CCV:00098'
    # Hard-rule for 'CCV:00099'
    if Evidence_type == 'ResidenceCertificate' and text.isdigit() and len(text) == 17: return 'CCV:00099'
    # Hard-rule for 'CCV:00100'
    if Evidence_type == 'ResidenceCertificate' and text.isdigit() and len(text) == 3: return 'CCV:00100'


    return None
