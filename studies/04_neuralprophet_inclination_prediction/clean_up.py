import os
import glob

# 1. Clean the LaTeX report
tex_path = 'report/neuralprophet_inclination_prediction_report.tex'
with open(tex_path, 'r') as f:
    lines = f.readlines()

new_lines = []
in_intro = True
for line in lines:
    new_lines.append(line)
    if r'\label{fig:on-structure-record}' in line:
        # keep the \end{figure} which is the next line
        break

# The next line should be \end{figure}
end_figure_idx = len(new_lines)
if end_figure_idx < len(lines):
    new_lines.append(lines[end_figure_idx])

# We also want to remove the Abstract and Verdict that I added, if the user only wants the Introduction.
# Wait, let's just write exactly the header and the introduction.
# The introduction starts at \section{Introduction}.
header = []
intro = []
in_abstract = False
for line in new_lines:
    if line.startswith(r'\begin{abstract}'):
        in_abstract = True
    elif line.startswith(r'\end{abstract}'):
        in_abstract = False
        continue
    
    if in_abstract:
        continue
        
    if line.startswith(r'\panel{Operational verdict}'):
        continue

    header.append(line)

new_content = "".join(header) + "\n\\end{document}\n"

with open(tex_path, 'w') as f:
    f.write(new_content)

# 2. Clean the Python script
py_path = 'neuralprophet_inclination_prediction_study.py'
with open(py_path, 'r') as f:
    py_lines = f.readlines()

new_py_lines = []
for line in py_lines:
    new_py_lines.append(line)
    if line.strip() == "plt.show()":
        # Check if we just generated NP_F01_on_structure_record
        if len(new_py_lines) > 2 and 'NP_F01_on_structure_record' in new_py_lines[-2]:
            break

with open(py_path, 'w') as f:
    f.writelines(new_py_lines)

# 3. Clean outputs
import shutil
for f in os.listdir('outputs'):
    if not f.startswith('NP_F01_on_structure_record'):
        path = os.path.join('outputs', f)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

