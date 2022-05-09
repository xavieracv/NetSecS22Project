# This script finds IDN domains and generates a csv

from googlesearch import search
import numpy as np



def main():
    tld = [
    ".中国",
    ".РФ",
    ".한국",
    ".台灣",
    ".рус",
    ".भारत",
    ".ЕЮ",
    "இந்தியா",
    ".みんな"
    ]

    l = []
    for item in tld:
        for i in search("site:"+item, num=10, stop=10, pause=2):
            l.append(i)
            print(i)


    narr = list(map(lambda x: [x,1], l))
    np.savetxt('datasets/IDN2.csv', narr, delimiter=",", fmt='%s')


if __name__=="__main__":
    main()
