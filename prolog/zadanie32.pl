ciagA(0, 1).
ciagA(1, 1).
ciagA(N, Wynik) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    ciagA(N1, W1),
    ciagA(N2, W2),
    Wynik is W1 + W2.

sumaA(0, 1).
sumaA(N, Suma) :-
    N > 0,
    N1 is N - 1,
    sumaA(N1, SumaCzesciowa),
    ciagA(N, WartoscElementu),
    Suma is SumaCzesciowa + WartoscElementu.

ciagB(0, 2).
ciagB(1, 2).
ciagB(2, 2).
ciagB(N, Wynik) :-
    N > 2,
    N1 is N - 1,
    N3 is N - 3,
    ciagB(N1, W1),
    ciagB(N3, W3),
    Wynik is W1 - W3.

sumaB(0, 2).
sumaB(N, Suma) :-
    N > 0,
    N1 is N - 1,
    sumaB(N1, SumaCzesciowa),
    ciagB(N, WartoscElementu),
    Suma is SumaCzesciowa + WartoscElementu.

ciagC(0, 2).
ciagC(1, 3).
ciagC(N, Wynik) :-
    N > 1,
    N2 is N - 2,
    ciagC(N2, W2),
    Wynik is W2 * N2.

sumaC(0, 2).
sumaC(N, Suma) :-
    N > 0,
    N1 is N - 1,
    sumaC(N1, SumaCzesciowa),
    ciagC(N, WartoscElementu),
    Suma is SumaCzesciowa + WartoscElementu.