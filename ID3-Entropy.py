import pandas as pd
import numpy as np
import math
from collections import Counter

# הגדרות צבעים ANSI
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
UNDERLINE = '\033[4m'

# קווי מפריד
THICK_SEPARATOR = f"{MAGENTA}{BOLD}" + "═" * 120 + f"{RESET}"
THIN_SEPARATOR = f"{CYAN}" + "─" * 80 + f"{RESET}"
BRANCH_SEPARATOR = f"{YELLOW}" + "┌" + "─" * 78 + "┐" + f"{RESET}"


class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, classification=None, depth=0):
        self.attribute = attribute 
        self.value = value 
        self.classification = classification  
        self.children = {}  
        self.depth = depth  
        self.is_leaf = False
        self.entropy = 0
        self.samples_count = 0
        self.samples_distribution = {}


def shorten_text(text, max_len=30):
    text = str(text)
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def calculate_entropy(series, verbose=False):
    if len(series) == 0:
        return 0

    counts = series.value_counts()
    total = len(series)
    entropy = 0

    if verbose:
        clean_counts = {k: int(v) for k, v in counts.items()} # Delete comment to display distribution
        #print(f"{WHITE}  התפלגות ערכי עמודת מטרה: {clean_counts}{RESET}")

    for val, count in counts.items():
        p = count / total
        part = -p * math.log2(p)
        if verbose:
            print(f"{DIM} - ({count}/{total}) * log₂({count}/{total}){RESET}", end="")
        entropy += part

    if verbose:
        print(f"{YELLOW} = {entropy:.4f}\n{RESET}")
    return entropy


def calculate_information_gain(df, attribute, target_column, target_entropy, verbose=False):
    if verbose:
        print(f"{BOLD}{BLUE}ניתוח עבור עמודה: {shorten_text(attribute)}{RESET}")
        values = df[attribute].unique()
        print(f"{WHITE}תכונות אפשריות: {[shorten_text(v) for v in values]}{RESET}")
        print(THICK_SEPARATOR)

    values = df[attribute].unique()
    conditional_entropy = 0
    entropy_parts = []

    for val in values:
        subset = df[df[attribute] == val]
        weight = len(subset) / len(df)
        if verbose:
            print(f"{CYAN}  תכונה: '{shorten_text(val)}' (משקל: {len(subset)}/{len(df)}){RESET}")
        sub_entropy = calculate_entropy(subset[target_column], verbose=verbose)
        weighted_entropy = weight * sub_entropy
        entropy_parts.append(f"{len(subset)}/{len(df)} * {sub_entropy:.4f}")
        conditional_entropy += weighted_entropy

    if verbose:
        parts_str = " + ".join(entropy_parts)
        print(f"{GREEN}{BOLD}Entropy:{RESET} {parts_str} {GREEN}{BOLD}= {conditional_entropy:.4f}{RESET}")

    info_gain = target_entropy - conditional_entropy
    if verbose:
        print(
            f"{RED}{BOLD}Info Gain:{RESET} {target_entropy:.4f} - {conditional_entropy:.4f} {RED}{BOLD}= {info_gain:.4f}\n{RESET}")

    return info_gain


def get_best_attribute(df, attributes, target_column, verbose=False):
    # target attribute entropy
    target_entropy = calculate_entropy(df[target_column])

    if verbose:
        #print(f"{BOLD}{MAGENTA}בחירת התכונה הטובה ביותר:{RESET}")
        #print(f"{BOLD}{CYAN}אנטרופיה של עמודת המטרה '{shorten_text(target_column)}': {target_entropy:.4f}{RESET}")
        print(THICK_SEPARATOR)

    best_gain = -1
    best_attribute = None
    gains = {}

    for attr in attributes:
        gain = calculate_information_gain(df, attr, target_column, target_entropy, verbose)
        gains[attr] = gain
        if gain > best_gain:
            best_gain = gain
            best_attribute = attr

        if verbose:
            print(THICK_SEPARATOR)

    if verbose:
        print(f"{BOLD}{GREEN}סיכום Information Gains:{RESET}")
        for attr, gain in sorted(gains.items(), key=lambda x: x[1], reverse=True):
            marker = "★" if attr == best_attribute else " "
            print(f"{GREEN}{marker} {shorten_text(attr)}: {gain:.4f}{RESET}")
        print(f"{BOLD}{RED}התכונה הנבחרת: {shorten_text(best_attribute)} (Gain: {best_gain:.4f}){RESET}")

    return best_attribute, best_gain


def is_pure(df, target_column):
    return len(df[target_column].unique()) == 1


def get_majority_class(df, target_column):
    return df[target_column].mode()[0]


def print_node_info(node, df, target_column, depth=0):
    indent = "  " * depth
    branch_char = "├─" if depth > 0 else "┌─"

    if node.is_leaf:
        distribution = {k: int(v) if hasattr(v, 'item') else v for k, v in df[target_column].value_counts().items()}
        total = len(df)
        print(f"{indent}{GREEN}{BOLD}{branch_char} 🍃 LEAF: {node.classification}{RESET}")
        print(f"{indent}   📊 דוגמאות: {total}, התפלגות: {distribution}")
        print(f"{indent}   ✅ סיווג מושלם: {len(distribution) == 1}")
    else:
        print(f"{indent}{BLUE}{BOLD}{branch_char} 🌳 BRANCH: {shorten_text(node.attribute)}{RESET}")
        print(f"{indent}   📊 דוגמאות: {len(df)}")


def categorize_branches_by_purity(df, best_attribute, target_column):
    values = df[best_attribute].unique()
    pure_branches = []
    impure_branches = []

    for value in values:
        subset = df[df[best_attribute] == value]
        if is_pure(subset, target_column):
            pure_branches.append((value, subset))
        else:
            impure_branches.append((value, subset))

    return pure_branches, impure_branches


def create_leaf_node(df, target_column, depth, value):
    node = DecisionTreeNode(depth=depth, value=value)
    node.classification = df[target_column].iloc[0]
    node.is_leaf = True
    node.samples_count = len(df)
    node.samples_distribution = dict(df[target_column].value_counts())
    node.entropy = 0  # pure branch - entropy is zero
    return node


def build_decision_tree(df, target_column, attributes, depth=0, max_depth=10, min_samples=1, parent_value=None):
    indent = "  " * depth
    print(f"\n{indent}{BOLD}{YELLOW}{'=' * 20} רמה {depth} {'=' * 20}{RESET}")

    if parent_value:
        print(f"{indent}{CYAN}🔍 בניית תת-עץ עבור ערך: '{shorten_text(parent_value)}'{RESET}")

    print(f"{indent}{WHITE}📈 גודל דאטה סט: {len(df)} דוגמאות{RESET}")
    print(f"{indent}{WHITE}🎯 תכונות זמינות: {[shorten_text(attr) for attr in attributes]}{RESET}")

    # create new node
    node = DecisionTreeNode(depth=depth, value=parent_value)
    node.samples_count = len(df)
    node.samples_distribution = dict(df[target_column].value_counts())
    node.entropy = calculate_entropy(df[target_column])

    # stop condition
    if is_pure(df, target_column):
        node.classification = df[target_column].iloc[0]
        node.is_leaf = True
        print(f"{indent}{GREEN}{BOLD}🎉 סיווג מושלם! כל הדוגמאות שייכות לקטגוריה: '{node.classification}'{RESET}")
        print_node_info(node, df, target_column, depth)
        return node

    if len(attributes) == 0 or depth >= max_depth or len(df) < min_samples:
        node.classification = get_majority_class(df, target_column)
        node.is_leaf = True
        reason = "אין תכונות נוספות" if len(
            attributes) == 0 else f"הגיע לעומק מקסימלי ({max_depth})" if depth >= max_depth else f"מעט מדי דוגמאות ({len(df)})"
        print(f"{indent}{YELLOW}🛑 עצירה: {reason}{RESET}")
        print(f"{indent}{RED}📊 סיווג לפי רוב: '{node.classification}'{RESET}")
        print_node_info(node, df, target_column, depth)
        return node

    # choose best attribute
    print(f"\n{indent}{BOLD}{MAGENTA}🔍 בחירת התכונה הטובה ביותר:{RESET}")
    best_attribute, best_gain = get_best_attribute(df, attributes, target_column, verbose=True)

    node.attribute = best_attribute
    print_node_info(node, df, target_column, depth)

    # split to pure and impure branches
    print(f"\n{indent}{BOLD}{CYAN}🌟 חלוקה לפי תכונה: '{shorten_text(best_attribute)}'{RESET}")
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    pure_branches, impure_branches = categorize_branches_by_purity(df, best_attribute, target_column)

    # summarize
    total_branches = len(pure_branches) + len(impure_branches)
    print(
        f"{indent}{WHITE}📊 סה\"כ תת-ענפים: {total_branches} (טהורים: {len(pure_branches)}, מעורבים: {len(impure_branches)}){RESET}")

    # process pure branches first
    branch_counter = 0
    if pure_branches:
        print(f"\n{indent}{BOLD}{GREEN}🍃 ענפים טהורים (לא דורשים חישובים נוספים):{RESET}")
        for value, subset in pure_branches:
            branch_counter += 1
            print(f"\n{indent}{BRANCH_SEPARATOR}")
            print(
                f"{indent}{GREEN}📋 ענף {branch_counter}/{total_branches}: {shorten_text(best_attribute)} = '{shorten_text(value)}' (טהור){RESET}")
            print(f"{indent}{WHITE}   📊 {len(subset)} דוגמאות - סיווג מוחלט: '{subset[target_column].iloc[0]}'{RESET}")

            child_node = create_leaf_node(subset, target_column, depth + 1, value)
            node.children[value] = child_node

    # process impure branches
    if impure_branches:
        print(f"\n{indent}{BOLD}{YELLOW}🔍 ענפים מעורבים (דורשים חישובים נוספים):{RESET}")
        print(f"{indent}{RED}(להסתכל על הטבלה ולחפש אם יש התפלגות ברורה!){RESET}")
        for value, subset in impure_branches:
            branch_counter += 1
            print(f"\n{indent}{BRANCH_SEPARATOR}")
            print(
                f"{indent}{YELLOW}📋 ענף {branch_counter}/{total_branches}: {shorten_text(best_attribute)} = '{shorten_text(value)}' (מעורב){RESET}")
            distribution = {k: int(v) for k, v in subset[target_column].value_counts().items()}
            print(f"{indent}{WHITE}   📊 {len(subset)} דוגמאות - התפלגות: {distribution}{RESET}")

            # recursive tree build
            child_node = build_decision_tree(
                subset, target_column, remaining_attributes,
                depth + 1, max_depth, min_samples, value
            )
            node.children[value] = child_node

    return node


def print_tree_summary(node, depth=0):
    indent = "  " * depth

    purity = "❌ מעורב"

    if node.is_leaf:
        purity = "✅ טהור" if len(node.samples_distribution) == 1 else "❌ מעורב"
        print(f"{indent}🍃 {node.classification} ({node.samples_count} דוגמאות) {purity}")
    else:
        print(f"{indent}🌳 {shorten_text(node.attribute)} ({node.samples_count} דוגמאות) {purity}")
        for value, child in node.children.items():
            print(f"{indent}├ {shorten_text(value)}:")
            print_tree_summary(child, depth + 1)


def predict_sample(node, sample):
    if node.is_leaf:
        return node.classification

    attribute_value = sample[node.attribute]
    if attribute_value in node.children:
        return predict_sample(node.children[attribute_value], sample)
    else:
        # predict by majority
        return node.classification if node.classification else "לא ידוע"


def clean_data_thoroughly(df):
    print(f"{YELLOW}🧹 מבצע ניקוי יסודי של הנתונים...{RESET}")

    original_shape = df.shape

    # 1. remove formatting in cells
    df.columns = df.columns.astype(str)  
    df.columns = [
        col.strip()
        .replace('\u200f', '') 
        .replace('\u202b', '')  
        .replace('\u202c', '')  
        .replace('\u200e', '')  
        .replace('\u202a', '') 
        .replace('\n', ' ')
        .replace('\r', ' ')
        .replace('\t', ' ')
        for col in df.columns
    ]

    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda x:
                                x.strip()
                                .replace('\u200f', '')
                                .replace('\u202b', '')
                                .replace('\u202c', '')
                                .replace('\u200e', '')
                                .replace('\u202a', '')
                                .replace('\n', ' ')
                                .replace('\r', ' ')
                                .replace('\t', ' ')
                                if isinstance(x, str) else str(x)
                                )

        # replace missing values
        df[col] = df[col].replace(['', ' ', 'nan', 'NaN', 'None'], 'לא ידוע')

    # remove blank row
    df = df.dropna(how='all')

    # remove blank col
    df = df.dropna(axis=1, how='all')

    print(f"{GREEN}✅ ניקוי הושלם: {original_shape} → {df.shape}{RESET}")

    return df

def validate_data_for_decision_tree(df, target_column):
    print(f"{CYAN}🔍 בודק תקינות הנתונים...{RESET}")

    issues = []
    warnings = []

    # check if target col exists
    if target_column not in df.columns:
        issues.append(f"עמודת היעד '{target_column}' לא קיימת!")
        return issues, warnings

    # check dataset conditions
    target_unique = df[target_column].nunique()
    if target_unique == 0:
        issues.append("עמודת היעד ריקה לחלוטין!")
    elif target_unique == 1:
        warnings.append(f"עמודת היעד כוללת רק ערך אחד: '{df[target_column].iloc[0]}'")

    if len(df) < 2:
        issues.append(f"יש רק {len(df)} שורות - צריך לפחות 2")

    feature_columns = [col for col in df.columns if col != target_column]
    if len(feature_columns) == 0:
        issues.append("אין עמודות תכונות (רק עמודת היעד)")

    # Look for unique values
    for col in df.columns:
        unique_vals = df[col].nunique()
        total_vals = len(df)

        if unique_vals == total_vals and col != target_column:
            warnings.append(f"עמודה '{col}' כוללת ערכים יחודיים לכל שורה (ייתכן שזה מזהה)")

        if unique_vals > total_vals * 0.8 and col != target_column:
            warnings.append(
                f"עמודה '{col}' כוללת {unique_vals} ערכים שונים מתוך {total_vals} - ייתכן שזה לא מתאים לעץ החלטה")

    # warnings
    if issues:
        print(f"{RED}❌ בעיות קריטיות שימנעו בניית עץ:{RESET}")
        for issue in issues:
            print(f"   • {issue}")

    if warnings:
        print(f"{YELLOW}⚠️  אזהרות (לא מונעות בניית עץ אבל כדאי לשים לב):{RESET}")
        for warning in warnings:
            print(f"   • {warning}")

    if not issues and not warnings:
        print(f"{GREEN}✅ כל הבדיקות עברו בהצלחה!{RESET}")

    return issues, warnings


def analyze_excel_and_build_tree(file_path, target_column, max_depth=10, min_samples=1):
    try:
        print(f"{BOLD}{CYAN}🚀 טוען קובץ Excel: {file_path}{RESET}")
        df = pd.read_excel(file_path, engine="openpyxl")

        # clean dataset
        df = clean_data_thoroughly(df)

        print(f"{GREEN}✅ נטען בהצלחה! {len(df)} שורות, {len(df.columns)} עמודות{RESET}")
        print(f"{WHITE}📋 עמודות: {list(df.columns)}{RESET}")
        print(f"{YELLOW}🎯 עמודת יעד: '{target_column}'{RESET}")

        # validate dataset
        issues, warnings = validate_data_for_decision_tree(df, target_column)

        if issues:
            print(f"{RED}❌ לא ניתן להמשיך בגלל בעיות קריטיות!{RESET}")
            return None, None

        # store attributes
        attributes = [col for col in df.columns if col != target_column]

        print(f"\n{BOLD}{MAGENTA}📊 סטטיסטיקות ראשוניות:{RESET}")
        print(f"{WHITE}🎯 התפלגות עמודת יעד '{target_column}':{RESET}")
        target_counts = df[target_column].value_counts()
        for val, count in target_counts.items():
            count_int = int(count) if hasattr(count, 'item') else count
            print(f"   {val}: {count_int} ({count_int / len(df) * 100:.1f}%)")

        initial_entropy = calculate_entropy(df[target_column])
        print(f"{YELLOW}📈 אנטרופיה ראשונית: {initial_entropy:.4f}{RESET}")

        print(f"\n{THICK_SEPARATOR}")
        print(f"{BOLD}{RED}🌳 מתחיל בניית עץ החלטה 🌳{RESET}")
        print(f"{THICK_SEPARATOR}")

        # build decision tree
        tree_root = build_decision_tree(df, target_column, attributes, max_depth=max_depth, min_samples=min_samples)

        print(f"\n{THICK_SEPARATOR}")
        print(f"{BOLD}{GREEN}🎉 עץ החלטה הושלם!{RESET}")
        print(f"{THICK_SEPARATOR}")

        print(f"\n{BOLD}{CYAN}📋 סיכום העץ:{RESET}")
        print_tree_summary(tree_root)

        return tree_root, df

    except FileNotFoundError:
        print(f"{RED}❌ שגיאה: הקובץ '{file_path}' לא נמצא!{RESET}")
        return None, None
    except Exception as e:
        print(f"{RED}❌ שגיאה בעת טעינת הקובץ: {str(e)}{RESET}")
        return None, None


def test_tree_predictions(tree_root, df, target_column, num_samples=5):
    print(f"\n{BOLD}{YELLOW}🧪 בדיקת חיזויים על {num_samples} דוגמאות אקראיות:{RESET}")

    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    correct = 0

    for i, idx in enumerate(sample_indices):
        sample = df.iloc[idx]
        actual = sample[target_column]
        predicted = predict_sample(tree_root, sample)
        is_correct = actual == predicted
        correct += is_correct

        status = f"{GREEN}✅ נכון{RESET}" if is_correct else f"{RED}❌ שגוי{RESET}"
        print(f"דוגמה {i + 1}: צפוי={predicted}, בפועל={actual} {status}")

    accuracy = correct / len(sample_indices) * 100
    print(f"\n{BOLD}{CYAN}📊 דיוק על הדוגמאות הנבדקות: {accuracy:.1f}% ({correct}/{len(sample_indices)}){RESET}")


if __name__ == "__main__":
    file_path = "TestData.xlsx"  # change to the appropriate filename
    target_column = "x" # change to your target col header 

    tree, df = analyze_excel_and_build_tree(file_path, target_column, max_depth=10, min_samples=1)

    if tree and df is not None:
        test_tree_predictions(tree, df, target_column, num_samples=5)

        print(f"\n{BOLD}{MAGENTA}🎯 העץ מוכן לשימוש! ניתן להשתמש בפונקציה predict_sample() לחיזוי דוגמאות חדשות.{RESET}")
    else:

        print(f"{RED}❌ בניית העץ נכשלה. בדוק את הקובץ ואת עמודת היעד.{RESET}")
