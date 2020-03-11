import random

_rand = random.Random(0)


def misspell(word, rand=None):
    if len(word) == 1:
        return word
    if rand is None:
        rand = _rand
    if len(word) < 4:
        mistakes = 1
    elif len(word) < 7:
        mistakes = 2
    else:
        mistakes = 3
    new_word = ""
    prev = 0
    for i in range(len(word)):
        char = word[i]
        if mistakes <= 0 or char in "1234567890":
            new_word += char
            prev = i
            continue
        mistake_type = rand.randint(1, 10)

        if mistake_type == 1:
            # delete
            pass
        if mistake_type == 2:
            # Swapping with prev
            new_word = new_word[:prev] + char + new_word[prev + 1 :] + word[prev]
        elif mistake_type == 3:
            # Alteration
            alter_char = chr(rand.choice(range(ord('a'), ord('z') + 1)))
            new_word += alter_char
        elif mistake_type == 4:
            new_word += char + char
        else:
            # no mistake
            new_word += char
            prev = i
            continue
        prev = i
        mistakes -= 1
    return new_word
