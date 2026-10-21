import os, re

repo_dir = r"D:\IISERB\Research\Thermoplasmonics\Thesis\github repository"
thesis_dir = r"D:\IISERB\Research\Thermoplasmonics\Thesis"

# 1. README fixes
readme_fnf_path = os.path.join(repo_dir, "Readme_fnf.md")
readme_old_path = os.path.join(repo_dir, "README.md")

if os.path.exists(readme_fnf_path):
    with open(readme_fnf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'\\operatorname\{([^}]+)\}', r'\\mathrm{\1}', content)
    content = content.replace(r'\vert{}', '|')
    content = content.replace(r'\|', '|')
    content = re.sub(r'(?i)\(Baffou-style\)', '', content)
    content = re.sub(r'(?i)Baffou-style', '', content)
    content = content.replace("BS-MS Thesis", "BS-MS Project Report")
    content = content.replace("full_thesis_report.pdf", "full_project_report.pdf")
    
    if os.path.exists(readme_old_path):
        os.remove(readme_old_path)
        
    with open(readme_old_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.remove(readme_fnf_path)

pdf_old = os.path.join(repo_dir, "docs", "full_thesis_report.pdf")
pdf_new = os.path.join(repo_dir, "docs", "full_project_report.pdf")
if os.path.exists(pdf_old):
    os.rename(pdf_old, pdf_new)

# 2. .tex files
tex_files = [os.path.join(thesis_dir, "main.tex"), os.path.join(thesis_dir, "short_report.tex")]
for root, _, files in os.walk(os.path.join(thesis_dir, "Chapters")):
    for f in files:
        if f.endswith(".tex"):
            tex_files.append(os.path.join(root, f))

for tf in tex_files:
    if os.path.exists(tf):
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(r'\bThesis\b', 'Project Report', content)
        content = re.sub(r'\bthesis\b(?!(\.cls))', 'project report', content)
        content = re.sub(r'\[PLACEHOLDER[^\]]*\]', '', content)
        content = re.sub(r'(?i)Baffou-style', '', content)
        content = re.sub(r'% =+ COPY FROM HERE =+\n?', '', content)
        content = re.sub(r'% =+ STOP COPYING HERE =+\n?', '', content)
        content = re.sub(r'We need to be honest here:.*?\n', '', content)
        content = re.sub(r'% This appendix provides.*?(?=\n\n)', '', content, flags=re.DOTALL)
        
        with open(tf, 'w', encoding='utf-8') as f:
            f.write(content)

# 3. Python files
for root, _, files in os.walk(os.path.join(repo_dir, "src")):
    for f in files:
        if f.endswith(".py"):
            pf = os.path.join(root, f)
            with open(pf, 'r', encoding='utf-8') as file_obj:
                content = file_obj.read()
            
            content = re.sub(r'(?i)#\s*CRITICAL FIX:.*?\n', '', content)
            content = re.sub(r'(?i)#\s*NEW LOGIC:.*?\n', '', content)
            content = re.sub(r'#\s*-{10,}\n', '', content)
            content = re.sub(r'(?i)#\s*We ONLY import.*?\n', '', content)
            content = re.sub(r'(?i)#\s*We will override.*?\n', '', content)
            content = re.sub(r'(?i)#\s*FIX \d+:.*?\n', '', content)
            
            with open(pf, 'w', encoding='utf-8') as file_obj:
                file_obj.write(content)
