from numpy import exp
def qq(word):
    nlines = 4227933.
    count = [0 for i in range(100)]
    for line in open('article_4227933_lines.txt', 'r'):
        c = [1 if word==w.strip() else 0 for w in line.split()]
        c = sum(c)
        if c < 100:
            count[c] += 1
    pdf = [c/nlines for c in count]
    cdf = [sum(pdf[:i]) for i in range(1, 101)]
    count_sum = sum([idx*c for idx, c in enumerate(count)])
    lam = nlines/count_sum
    epdf = [lam*exp(-lam*i) for i in range(100)]
    ecdf = [1-exp(-lam*i) for i in range(100)]
    return pdf, cdf, epdf, ecdf

