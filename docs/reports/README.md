# Reports

The interim report (`EdgeVLM-Drive_Interim_Report_Phase1.pdf` / `.docx`) is kept in this folder locally
but is **excluded from Git** (see `.gitignore`) because its title block lists student ID numbers and
university e-mail addresses, and anything pushed to a public repository stays in its history.

To publish the report: export a copy without those details (or keep them if every author agrees),
then add it explicitly:

```bash
git add -f docs/reports/<file>.pdf
```
