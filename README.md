A feladat a következő alap képmanipulációs rutin implementálása volt.

* Negálás
* Gamma transzformáció
* Logaritmikus transzformáció
* Szürkítés
* Hisztogram készítés
* Hisztogram kiegyenlítés
* Átlagoló szűrő (Box szűrő)
* Gauss szűrő
* Sobel éldetektor
* Laplace éldetektor
* Jellemzőpontok detektálása

Ezeket a rutinokat cudában implementáltam majd egy dll-re fordítottam le őket.
Az implementálás során igyekeztem a kerneleket úgy kialakítani hogy ne kelljen az egyes részeket többször implementálni mint béldául a szürkítést vagy egy maszk végig futtatását a képen.

A dll-t egy c# wpf alapú alkalmazásból hívom meg ami a képet és hozzá tartozó adatokat adja át
a képet byte sorként és a kép szélességét és magasságát illetve azt hogy a kép hány byte-al egészít ki egy sort. 

Van lehetőség paramétert beadni bizonyos helyeken ezt a középen lévő textbox-ban lehet de van beállítva mindenhova alap értelmezett érték ami akkor kerül be ha a beírt érték nem megfelelő formátumu vagy esetleg nincs is.
