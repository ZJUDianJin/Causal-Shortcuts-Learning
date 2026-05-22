import os
import json
import re
import sys
import tiktoken
from fractions import Fraction

def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def clean_token_text(s: str) -> str:
    
    if s is None:
        return ''
    s = str(s)
    s = s.replace('<|endoftext|>', '')
    
    s = re.sub(r'\\\(|\\\)|\$\$|\$|\\\[|\\\]', '', s)
    
    s = re.sub(r'\\text\{(.*?)\}', r'\1', s)
    s = re.sub(r'text{', '', s)
    
    
    s = s.replace('\\', '')
    
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_expr(expr: str) -> str:
    
    expr = clean_token_text(expr)
    expr = expr.strip()
    
    expr = expr.replace(r'\times', '*').replace(r'\cdot', '*')
    expr = expr.replace(r'\ ', '').replace(' ', '')
    
    expr = re.sub(r'frac\{(\d+)\}\{(\d+)\}', lambda m: f"{m.group(1)}/{m.group(2)}", expr)
    
    try:
        if '/' in expr:
            val = Fraction(expr)
            expr = str(float(val))
        else:
            val = float(expr)
            expr = str(val)
    except Exception:
        
        expr = re.sub(r'\s+', '', expr).lower()
    return expr

def normalize_text_for_compare(s: str) -> str:
    
    s = clean_token_text(s)
    
    s = s.lower()
    
    s = s.strip(" \t\n\r:：。.、,，;；\"'`")
    
    s = re.sub(r'\s+', ' ', s)
    return s

def parse_options_from_question(question_text: str) -> dict:
    
    text = question_text or ''
    text = text.replace('\r', '\n')

    
    
    
    
    
    marker_re = re.compile(
        r'(?im)(?:^|\n|##Choices\s*)(?:\(([A-Da-d])\)|([A-Da-d]))[\.\)\：:\uFF0E\uFF09\uFF1A]?\s*'
    )
    markers = list(marker_re.finditer(text))
    options = {}

    if markers:
        for i, m in enumerate(markers):
            label = (m.group(1) or m.group(2)).upper()
            start = m.end()
            end = markers[i+1].start() if i + 1 < len(markers) else len(text)
            content = text[start:end].strip()
            content = clean_token_text(content)
            options[label] = content
    else:
        
        inline_re = re.findall(r'([A-Da-d])\s*[\.\)\：:]\s*([^A-Da-d]+)', text)
        for label, content in inline_re:
            label = label.upper()
            content = clean_token_text(content)
            options[label] = content

    return options


def parse_generation_answer(gen_text: str, question_text: str, ground_truth: str):
    
    if gen_text is None:
        return None
    gen_raw = str(gen_text)
    gen = clean_token_text(gen_raw)

    
    options = parse_options_from_question(question_text)
    # print(options)
    
    gt = (ground_truth or '').strip()
    gt = re.sub(r'^[\.\s]*', '', gt)  
    gt = gt.upper()
    
    if re.fullmatch(r'[A-D]', gt):
        gt_label = gt
        gt_option_text = options.get(gt_label, '')  

    else:
        
        gt_label = None
        gt_option_text = ground_truth.strip()

    
    gt_option_text_norm = normalize_text_for_compare(gt_option_text) if gt_option_text else ''
    
    gt_option_expr_norm = normalize_expr(gt_option_text) if gt_option_text else ''

    
    boxed = re.findall(r'boxed\{(.*?)\}', gen_raw)
    for box in boxed:
        box_clean = clean_token_text(box)
        
        
        if re.fullmatch(r'^[A-Da-d][\.\)\s]*$', box_clean.strip()):
            label = re.sub(r'[\.\)\s]+$', '', box_clean.strip()).upper()
            
            return label
        
        if gt_option_text:
            
            box_expr_norm = normalize_expr(box_clean)
            # print(f"box:{box_expr_norm},gt_option_expr_norm:{gt_option_expr_norm}")
            
            
            try:
                if float(box_expr_norm) == float(gt_option_expr_norm):
                    return gt_label if gt_label else gt_label_from_options(options, gt_option_text)
            except Exception:
                pass
            
            box_txt_norm = normalize_text_for_compare(box_clean)
            if gt_option_text_norm and (box_txt_norm == gt_option_text_norm or gt_option_text_norm in box_txt_norm or box_txt_norm in gt_option_text_norm):
                return gt_label if gt_label else gt_label_from_options(options, gt_option_text)
        
        m = re.match(r'^\s*([A-Da-d])\s*[\.\)\：:]\s*(.*)$', box_clean, flags=re.S)
        if m:
            label = m.group(1).upper()
            opt = clean_token_text(m.group(2))
            
            if gt_option_text:
                if is_text_equivalent(opt, gt_option_text):
                    return label
            else:
                
                pass

    
    ans_patterns = [
        r'(?:答案是|答案为|正确答案是|正确答案为|answer is|the answer is|ans(?:wer)?[:：]?)\s*([^\n\r<]+)',
        r'选择[:：]?\s*([A-Da-d])',
        r'选为[:：]?\s*([A-Da-d])'
    ]
    for pat in ans_patterns:
        m = re.search(pat, gen, flags=re.I)
        if m:
            candidate = clean_token_text(m.group(1))
            
            
            lab_m = re.match(r'^([A-Da-d])\b', candidate)
            if lab_m:
                return lab_m.group(1).upper()
            
            if gt_option_text and is_text_equivalent(candidate, gt_option_text):
                return gt_label if gt_label else gt_label_from_options(options, gt_option_text)

    
    lines = [l.strip() for l in gen.splitlines() if l.strip()]
    if lines:
        first = clean_token_text(lines[0])
        
        m = re.match(r'^([A-Da-d])[\.\)\：:]*\s*(.*)$', first, flags=re.S)
        if m:
            label = m.group(1).upper()
            trailing = clean_token_text(m.group(2))
            if trailing == '':
                return label
            
            if gt_option_text and is_text_equivalent(trailing, gt_option_text):
                return label
        
        if gt_option_text and is_text_equivalent(first, gt_option_text):
            return gt_label if gt_label else gt_label_from_options(options, gt_option_text)

    return None


def is_text_equivalent(a: str, b: str) -> bool:
    
    if a is None or b is None:
        return False
    a = clean_token_text(a)
    b = clean_token_text(b)
    if not a or not b:
        return False
    
    try:
        a_expr = normalize_expr(a)
        b_expr = normalize_expr(b)
        
        
        if a_expr == b_expr:
            return True
        
        try:
            af = float(a_expr); bf = float(b_expr)
            if abs(af - bf) < 1e-9:
                return True
        except Exception:
            pass
    except Exception:
        pass
    
    an = normalize_text_for_compare(a)
    bn = normalize_text_for_compare(b)
    if an == bn:
        return True
    
    if an and bn and (an in bn or bn in an):
        return True
    return False

def gt_label_from_options(options: dict, gt_option_text: str):
    
    for label, text in options.items():
        if is_text_equivalent(text, gt_option_text):
            return label
    return None


def evaluate_folder_(folder_path):
    total = 0
    correct = 0

    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                generations_list = data.get('generations', [])

                for item in generations_list:
                    question_text = item.get('question', '') or ''
                    gen_answer = item.get('generations', '') or ''
                    ground_truth = (item.get('ground_truth', '') or '').strip()

                    total += 1
                    predicted = parse_generation_answer(gen_answer, question_text, ground_truth)

                    if predicted == (ground_truth.strip().upper() if re.fullmatch(r'[A-Da-d]', ground_truth.strip(), flags=re.I) else predicted):
                        
                        
                        correct += 1
                    else:
                        
                        print("---- 文件:", filename)
                        print("====question_text====\n", question_text)
                        print("===generate raw===\n", str(gen_answer).replace("<|endoftext|>", ""))
                        print("===ground_truth===\n", ground_truth)
                        print("===predicted===\n", predicted)
                        print("---------------------------------\n")

    accuracy = correct / total if total > 0 else 0
    return accuracy



def evaluate_folder(folder_path):
    acc_list = []
    filename_list = []
    token_list = []
    avg_token_list = []

    file_list = sorted(os.listdir(folder_path))
    

    for filename in file_list:
        if not filename.endswith('.json'):
            continue
        if not filename.startswith('sat'):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = 0
        correct = 0
        total_effective_tokens = 0

        generations_list = data.get('generations', [])
        for item in generations_list:
            question_text = item.get('question', '') or ''
            gen_answer = item.get('generations', '') or ''
            ground_truth = (item.get('ground_truth', '') or '').strip()

            total += 1
            predicted = parse_generation_answer(gen_answer, question_text, ground_truth)
            
            effective_tokens = count_effective_tokens(gen_answer)
            total_effective_tokens += effective_tokens

            if re.fullmatch(r'[A-Da-d1-4]', ground_truth.strip(), flags=re.I):
                target = ground_truth.strip().upper()
                if predicted == target:
                    correct += 1
            else:
                if predicted is not None:
                    options = parse_options_from_question(question_text)
                    pred_text = options.get(predicted, "")
                    if is_text_equivalent(pred_text, ground_truth):
                        correct += 1

        accuracy = correct / total if total > 0 else 0.0
        acc_list.append(accuracy)
        filename_list.append(filename)
    
        token_list.append(total_effective_tokens)
        avg_tk = total_effective_tokens / total if total > 0 else 0.0
        avg_token_list.append(avg_tk)

    return acc_list, filename_list, token_list, avg_token_list

def aggregate_results(folder_path):
    acc_list, filename_list, token_list, avg_token_list = evaluate_folder(folder_path)

    print("=" * 110)
    print("Evaluation Results".center(110))
    print("=" * 110)
    print(f"{'Filename':<57} {'Accuracy':<12} {'Total Tokens':<12} {'Avg Tokens':<12}")
    print("-" * 110)

    for name, acc, tk, avg_tk in zip(filename_list, acc_list, token_list, avg_token_list):
        print(f"{name:<60} {acc:<12.2%} {tk:<12} {avg_tk:<12.1f}")

    print()


if __name__ == "__main__":
    aggregate_results("")
