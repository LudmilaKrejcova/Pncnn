# První trénování pneumatik v pNCNN

Rozbalte obsah ZIPu přímo do C:\Users\bcRaptor\Projects\pncnn, vedle main.py.
Nové soubory trainTires.py a tireNconv.py musí být vedle složky common.
Původní soubory projektu není potřeba přepisovat. Používá se common/unet.py z dodaného projektu.

V PowerShellu ve složce projektu nastavte:

```powershell
$data = 'C:\Users\bcRaptor\Desktop\FIT\diplomka\MiniPrograms\sickPrepared\tilesPilot'
```

Ve složce dat musí být původní tiles.jsonl, normalization.json a adresářová struktura s NPZ.
Není potřeba přesouvat data do repozitáře ani instalovat jinou verzi PyTorch.

## 1. Krátká kontrola

```powershell
.\.venv\Scripts\python.exe trainTires.py --data "$data" --output runs\tireSmoke --epochs 2 --train-limit 8 --val-limit 8 --batch-size 1
```

Po prvním batchi se vypíše loss a čas. Prohlédněte runs\tireSmoke\epoch002.png.
Tohle ověřuje technický běh, ne kvalitu modelu. Loss může být i záporný, protože jde o pravděpodobnostní loss.

## 2. Pilotní trénink na všech připravených train/val dlaždicích

```powershell
.\.venv\Scripts\python.exe trainTires.py --data "$data" --output runs\tirePilot --epochs 3 --batch-size 1
```

Trénuje od začátku, nepoužívá předtrénované KITTI/NYU váhy ani kontrolní běh tireSmoke.
Rozdělení čte z tiles.jsonl. Testovací pneumatiky nepoužívá při trénování ani výběru modelu.
Počet dlaždic vypíše na začátku. Délku běhu odhadněte až podle rychlosti prvních 25 batchů.
Notebook připojte k napájení a po dobu běhu vypněte automatické uspání.

## Výsledky

- last.pt: poslední dokončená epocha, model, optimizer, normalizace a stav generátoru PyTorch.
- best.pt: nejlepší model podle validačního MAE v dírách; pokud ve výběru nejsou díry, podle MAE všech validních cílů.
- metrics.csv: loss, MAE a RMSE v mm pro všechny platné cíle a samostatně pro díry; také počty pixelů.
- epochNNN.png: první validační dlaždice, vstup, predikce, prozatímní GT, maska děr a chyba v dírách.
- settings.json: nastavení a počty dlaždic.

Nejdůležitější sloupce: val_holesMaeMm a val_holesRmseMm. Menší je lepší.
Hodnoty jsou měřené vůči prozatímnímu GT z fúze; nejde o nezávislé přesné měření.
Obrázek ukazuje vždy stejnou první validační dlaždici; nemusí být reprezentativní pro všechny pneumatiky.

## Pokračování a přesun na jiný počítač

```powershell
.\.venv\Scripts\python.exe trainTires.py --data "$data" --output runs\tirePilot --epochs 10 --batch-size 1 --resume runs\tirePilot\last.pt
```

--epochs je CELKOVÝ cílový počet epoch. Po dokončení tří epoch pokračuje čtvrtou až desátou.
Ctrl+C přeruší běh; pokračovat lze od poslední dokončené epochy, nikoli od posledního batchu.
Pro pokračování kontrolního běhu zachovejte --train-limit 8 --val-limit 8 a stejný seed.
Používejte pouze vlastní nebo důvěryhodné checkpointy.

Na výpočetní stroj přeneste projekt včetně obou nových .py souborů, celou složku tilesPilot a runs/tirePilot.
Windows .venv nepřenášejte. Prostředí a způsob spuštění jobu je potřeba nastavit podle cílového stroje.
Na Linuxu bude příkaz například:

```bash
python trainTires.py --data /path/to/tilesPilot --output runs/tirePilot --epochs 10 --batch-size 1 --resume runs/tirePilot/last.pt
```

Cesty v tiles.jsonl jsou relativní, původní Windows sourceFiles se pro trénování neotevírají.
Skript hlídá stejnou normalizaci a stejný seznam trénovacích/validačních dlaždic při resume.
Pro tilesFull založte nový experiment; přímé resume s jiným výběrem dat skript záměrně odmítne.

## Co se změnilo proti původnímu pNCNN

Jde o adaptaci pNCNN pro maskované mapy pneumatik. Zachovává UNetSP, NConvUNet a pravděpodobnostní loss
precision * squaredError - log(precision). Confidence estimator má dva vstupy: normalizované Z a masku.
Počáteční confidence je mimo měřené body nulová. Intenzita, X ani druhá kamera nevstupují do modelu.
Hloubky se transformují trénovacím mean/scale z normalization.json. Platnost se nikdy neurčuje znaménkem Z.
Loss a metriky používají targetValidMask, vylučují padding a odmítají nejisté zdrojové kategorie.
Loss se učí ze všech důvěryhodných cílových pixelů; díry mají samostatné metriky, nikoli vyšší váhu v loss.

Validační metriky sčítají absolutní/kvadratické chyby a počty pixelů, ne průměry dlaždic.
Překrývající se validační/testovací dlaždice ve stejném skenu a kameře se odmítnou; pro ně by bylo potřeba
nejprve rekonstruovat celý sken. Dvě vstupní kamery jsou dvě samostatné predikční úlohy.
Překryvy trénovacích dlaždic jsou povolené; train metriky jsou proto pouze diagnostika optimalizace.

## Ověření

Testováno s PyTorch 2.14.1 CPU, nikoli přímo na Windows / GTX 1650 / PyTorch 1.13.1.
Použité API je zvolené s ohledem na dodaný PyTorch 1.13.1.
Na dodané 256x256 dlaždici prošel forward, backward a 12 optimalizačních kroků; loss klesl z 0.60446 na 0.40584.
Prošla kontrola ignorování neplatných cílů a shody predikce po uložení/načtení vah.
Celý train/val běh, obrázek a resume byly zkoušeny na technické fixture s kopií stejné dlaždice:
čísla z této fixture nejsou validační výsledky a nelze jimi dokládat generalizaci.
