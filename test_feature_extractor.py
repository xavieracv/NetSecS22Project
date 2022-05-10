import feature_extractor

def test_vowels_consonant_ratio():
    domain = "cA3lo.Kr5-slTp1ndi#op0"
    num_vowels = 4
    num_consonants = 11
    ratio = num_vowels / num_consonants
    assert(feature_extractor.vowel_consonant_ratio(domain) == ratio)

def test_num_digits():
    domain = "c#$gEpfLX@fm9p.0DP36#4"
    num_digits = 5
    assert(feature_extractor.num_digits(domain) == num_digits)

def test_num_special():
    domain = "3#D0X&*#432.0@ncO:OS-d.sA%"
    num_special = 10
    assert(feature_extractor.num_special(domain) == num_special)

def test_num_colons():
    domain = "3feoimwc[:v4tv:cpalmwrf:"
    num_colons = 3
    assert(feature_extractor.num_colons(domain) == num_colons)

    domain2 = "www.google.com"
    num_colons2 = 0
    assert(feature_extractor.num_colons(domain2) == num_colons2) 
        
def test_num_dashes():
    domain = "he1l#-w0rld.how-4re-you-do1ng"
    num_dashes = 4
    assert(feature_extractor.num_dashes(domain) == num_dashes)

def test_shannon_entropy():
    domain = "www.google.com"
    assert(feature_extractor.shannon_entropy(domain) - 2.84237 < 0.0001)

if __name__ == "__main__":
    test_vowels_consonant_ratio()
    test_num_digits()
    test_num_special()
    test_num_colons()
    test_num_dashes()
    test_shannon_entropy()
    print("All tests passed!")