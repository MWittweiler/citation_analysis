# Citation Analysis Setup Guide

A beginner-friendly setup guide for running the project at:

`https://github.com/MWittweiler/citation_analysis`

This guide is written for people who may never have installed Python before. It includes:

- a **Windows** version
- a **macOS** version
- a **first test run with the sample files** from the repository
- a final section on how to add **additional Latin texts** from the Tesserae corpus

---

## What this guide helps you do

By the end, you should be able to:

1. install Python on your computer
2. install Git
3. download the project repository
4. create a project-specific Python environment
5. install all required packages
6. run the script with the sample texts included in the repository
7. optionally add more texts from the Tesserae corpus

---

## What you need before you start

You need:

- an internet connection
- permission to install software on your computer
- enough free disk space for Python packages and language models
- some patience during the first installation, because a few required packages are large

### Disk space estimate

To be comfortable, plan for **at least 6 GB of free disk space**.

A safer recommendation is **8-10 GB free** so you have room for:

- Python itself
- Git
- the project folder
- the virtual environment
- downloaded package files and caches
- the Latin language model used by the project
- result files you create later

If your computer has less than that, the installation may still work, but you are more likely to run into space problems during package installation.

---

# Part 1: Windows

## 1. Install Python

Open this official Python download page for Windows in your browser:

`https://www.python.org/downloads/windows/`

Then:

1. Download a **Python 3.12** installer for Windows.
2. Run the installer.
3. **Important:** before clicking Install, make sure you check the box:
   - **Add Python to PATH**
4. Finish the installation.

### Check that Python works

### How to open PowerShell

Use one of these methods:

- press the **Windows key**, type **PowerShell**, then press **Enter**
- or right-click the **Start** button and choose **Windows PowerShell** or **Terminal**
- or open **VS Code**, then open **Terminal** from the menu

Now **run in PowerShell**:

```powershell
python --version
```

If that does not work, **run in PowerShell**:

```powershell
py --version
```

You should see a Python version number.

---

## 2. Install Git

Open this official Git download page for Windows in your browser:

`https://git-scm.com/download/win`

Then:

1. Download **Git for Windows**.
2. Run the installer.
3. The default installation options are usually fine.

### Check that Git works

**Run in PowerShell**:

```powershell
git --version
```

You should see a Git version number.

---

## 3. Choose where the project should live

For example, **run in PowerShell**:

```powershell
cd $HOME\Documents
```

This moves you into your Documents folder.

---

## 4. Download the project

**Run in PowerShell**:

```powershell
git clone https://github.com/MWittweiler/citation_analysis.git
cd citation_analysis
```

This creates a local copy of the project and moves you into it.

---

## 5. Create a virtual environment

A virtual environment is a project-specific Python environment. It keeps this project separate from other Python projects on your computer.

**Run in PowerShell**:

```powershell
py -3.12 -m venv ca_env
```

If that does not work, **run in PowerShell**:

```powershell
python -m venv ca_env
```

---

## 6. Activate the virtual environment

**Run in PowerShell**:

```powershell
.\ca_env\Scripts\Activate.ps1
```

If PowerShell blocks this, **run in PowerShell**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ca_env\Scripts\Activate.ps1
```

After activation, you should see something like `(ca_env)` at the beginning of the line.

---

## 7. Upgrade pip and installation tools

**Run in PowerShell**:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 8. Install the required packages

Make sure you are still in the `citation_analysis` folder and that the virtual environment is active.

Then **run in PowerShell**:

```powershell
pip install -r requirements.txt
```

This may take a while the first time.

---

## 9. Run the sample analysis

The repository already contains sample files. To test that everything works, **run in PowerShell**:

```powershell
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose
```

This is the full sample run.

### What happens next

- the script reads the two sample texts
- it analyzes possible citations or parallels
- it writes the results to an Excel file

The sample run may take a while.

---

## 10. Faster test run for beginners

If you want a quicker first test, **run in PowerShell**:

```powershell
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose --complura-only
```

This is useful if you want to make sure the installation worked before running the larger analysis.

---

## 11. When you are done

To leave the virtual environment, **run in PowerShell**:

```powershell
deactivate
```

The next time you want to use the project, go back into the project folder and activate the environment again.

**Run in PowerShell**:

```powershell
cd $HOME\Documents\citation_analysis
.\ca_env\Scripts\Activate.ps1
```

---

# Part 2: macOS

## 1. Install Python

Open the official Python downloads page in your browser:

`https://www.python.org/downloads/`

Then:

1. Download a **Python 3.12** installer for macOS.
2. Run the installer.
3. Finish the installation.

### Check that Python works

Open **Terminal**.

A simple way is:

- press **Command + Space**
- type **Terminal**
- press **Enter**

Then **run in Terminal**:

```bash
python3 --version
```

You should see a Python version number.

---

## 2. Install Git

Open the official Git download page in your browser:

`https://git-scm.com/download/mac`

Many macOS computers already have Git or offer to install it automatically when needed. To check, **run in Terminal**:

```bash
git --version
```

If Git is not installed, you can install it from the official Git website.

If you already use Homebrew, you can also **run in Terminal**:

```bash
brew install git
```

---

## 3. Choose where the project should live

For example, **run in Terminal**:

```bash
cd ~/Documents
```

---

## 4. Download the project

**Run in Terminal**:

```bash
git clone https://github.com/MWittweiler/citation_analysis.git
cd citation_analysis
```

---

## 5. Create a virtual environment

**Run in Terminal**:

```bash
python3.12 -m venv ca_env
```

If that does not work, try **run in Terminal**:

```bash
python3 -m venv ca_env
```

---

## 6. Activate the virtual environment

**Run in Terminal**:

```bash
source ca_env/bin/activate
```

After activation, you should see `(ca_env)` at the beginning of the terminal line.

---

## 7. Upgrade pip and installation tools

**Run in Terminal**:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 8. Install the required packages

**Run in Terminal**:

```bash
pip install -r requirements.txt
```

This may take a while the first time.

---

## 9. Run the sample analysis

**Run in Terminal**:

```bash
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose
```

### What happens next

- the script reads the two sample texts
- it searches for citation-like relationships
- it writes the results to an Excel file

---

## 10. Faster first test

If you want to test the setup more quickly, **run in Terminal**:

```bash
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose --complura-only
```

---

## 11. When you are done

To leave the virtual environment, **run in Terminal**:

```bash
deactivate
```

The next time you want to use the project, **run in Terminal**:

```bash
cd ~/Documents/citation_analysis
source ca_env/bin/activate
```

---

# Part 3: How to know the installation worked

A successful run usually means:

- the script starts without an import error
- the program processes the sample files
- an Excel results file appears in the project folder

If that happens, your setup is ready.

---

# Part 4: Troubleshooting

## “python” is not recognized

This usually means Python was not added to your PATH during installation.

### Fix on Windows

Reinstall Python and make sure **Add Python to PATH** is checked.

### Fix on macOS

Try `python3` instead of `python`.

---

## Git is not recognized

Install Git and open a new terminal window.

---

## The virtual environment will not activate on Windows

**Run in PowerShell**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ca_env\Scripts\Activate.ps1
```

---

## `pip install -r requirements.txt` takes a long time

That is normal. The project needs several large packages and language resources.

---

## The full sample run takes a long time

That is also normal. For a quicker first test, use the `--complura-only` option.

---

# Part 5: Adding more texts from Tesserae

If you want to analyze more texts than the ones already provided in the repository, you can use Latin texts from the Tesserae corpus here:

`https://github.com/tesserae/tesserae/tree/master/texts/la`

These files are already close to the format needed by the script, which makes them a good source for additional material.

---

## 1. What format does the script need?

The project expects **tab-separated text files** with:

- a left column containing metadata or an identifier
- a right column containing the actual text

The Tesserae `.tess` files are usually very close to this workflow because they already contain text segmented line by line. To make them easy to use here, the safest approach is to convert them into a simple two-column tab-separated format.

---

## 2. Create a folder for extra texts

Inside your `citation_analysis` folder, create a folder called `custom_texts`.

### Windows

**Run in PowerShell**:

```powershell
mkdir custom_texts
```

### macOS

**Run in Terminal**:

```bash
mkdir custom_texts
```

---

## 3. Download a Tesserae Latin text

Go to the Tesserae Latin texts folder in your browser and choose a text.

Save the `.tess` file into your `custom_texts` folder.

Example names might look like:

- `vergil.aeneid.part.1.tess`
- `lucan.bellum_civile.part.1.tess`

---

## 4. Convert the Tesserae file into the expected format

Create a new file named `convert_tesserae.py` in your main `citation_analysis` folder and paste this code into it:

```python
from pathlib import Path
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_tesserae.py input.tess output.txt")
    raise SystemExit(1)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with input_path.open("r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

with output_path.open("w", encoding="utf-8", newline="") as f:
    for i, line in enumerate(lines, start=1):
        f.write(f"{i}\t{line}\n")

print(f"Wrote {len(lines)} lines to {output_path}")
```

What this script does:

- reads the Tesserae text line by line
- removes empty lines
- adds a simple line number as the left column
- writes a tab-separated file that is easy to use with `citation_analysis.py`

---

## 5. Run the converter

### Windows example

**Run in PowerShell**:

```powershell
python convert_tesserae.py custom_texts\lucan.bellum_civile.part.1.tess custom_texts\lucan_bellum_civile_part1.txt
```

### macOS example

**Run in Terminal**:

```bash
python convert_tesserae.py custom_texts/lucan.bellum_civile.part.1.tess custom_texts/lucan_bellum_civile_part1.txt
```

After this, you should have a new file in the correct format.

---

## 6. Use your new text in the analysis

Once the file has been converted, you can use it as an input file just like the sample texts.

**Run in Terminal or PowerShell**:

```bash
python citation_analysis.py --input_1 custom_texts/lucan_bellum_civile_part1.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose
```

Make sure the genres match the texts you are using:

- `poetry`
- `prose`

---

## 7. Easy recommendation for beginners

If this is your first time using the project, use this order:

1. install Python
2. install Git
3. download the repository
4. create and activate the virtual environment
5. install the requirements
6. run the quick sample test with `--complura-only`
7. run the full sample analysis
8. only after that, add extra texts from Tesserae

This makes it much easier to tell whether a later problem comes from the original setup or from a newly added text.

---

# Part 6: Quick command summary

## Windows

**Run in PowerShell**:

```powershell
git clone https://github.com/MWittweiler/citation_analysis.git
cd citation_analysis
py -3.12 -m venv ca_env
.\ca_env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose --complura-only
```

## macOS

**Run in Terminal**:

```bash
git clone https://github.com/MWittweiler/citation_analysis.git
cd citation_analysis
python3.12 -m venv ca_env
source ca_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python citation_analysis.py --input_1 data/aeneid.txt --input_2 data/hieronymus_epistles.txt --genre_1 poetry --genre_2 prose --complura-only
```

---

# Final note

Once the sample files run successfully, your installation is complete. After that, you can start experimenting with your own texts.
