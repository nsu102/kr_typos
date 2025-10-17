import json
import random
import argparse
from typing import List, Dict, Tuple, Set, Optional
import re
import copy

# 한글 유니코드 상수
CHOSUNG_BASE = 0x1100
JUNGSUNG_BASE = 0x1161
JONGSUNG_BASE = 0x11A8
HANGUL_BASE = 0xAC00

# 초성, 중성, 종성 리스트
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 유사 자모 매핑
SIMILAR_CHOSUNG = {
    'ㄱ': ['ㄲ', 'ㅋ'],
    'ㄲ': ['ㄱ', 'ㅋ'],
    'ㄴ': ['ㄹ', 'ㅁ'],
    'ㄷ': ['ㄸ', 'ㅌ', 'ㄹ'],
    'ㄸ': ['ㄷ', 'ㅌ'],
    'ㄹ': ['ㄴ', 'ㄷ'],
    'ㅁ': ['ㄴ', 'ㅂ'],
    'ㅂ': ['ㅃ', 'ㅍ', 'ㅁ'],
    'ㅃ': ['ㅂ', 'ㅍ'],
    'ㅅ': ['ㅆ', 'ㅈ'],
    'ㅆ': ['ㅅ', 'ㅈ'],
    'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ'],
    'ㅉ': ['ㅈ', 'ㅊ'],
    'ㅊ': ['ㅈ', 'ㅉ'],
    'ㅋ': ['ㄱ', 'ㄲ'],
    'ㅌ': ['ㄷ', 'ㄸ'],
    'ㅍ': ['ㅂ', 'ㅃ'],
    'ㅎ': ['ㅇ'],
    'ㅇ': ['ㅎ']
}

SIMILAR_JUNGSUNG = {
    'ㅏ': ['ㅓ', 'ㅑ'],
    'ㅐ': ['ㅔ', 'ㅒ', 'ㅖ'],
    'ㅑ': ['ㅏ', 'ㅕ'],
    'ㅒ': ['ㅐ', 'ㅖ'],
    'ㅓ': ['ㅏ', 'ㅕ'],
    'ㅔ': ['ㅐ', 'ㅖ'],
    'ㅕ': ['ㅓ', 'ㅑ'],
    'ㅖ': ['ㅔ', 'ㅐ', 'ㅒ'],
    'ㅗ': ['ㅜ', 'ㅛ'],
    'ㅘ': ['ㅙ', 'ㅝ'],
    'ㅙ': ['ㅘ', 'ㅞ'],
    'ㅚ': ['ㅟ', 'ㅗ'],
    'ㅛ': ['ㅗ', 'ㅠ'],
    'ㅜ': ['ㅗ', 'ㅠ'],
    'ㅝ': ['ㅘ', 'ㅞ'],
    'ㅞ': ['ㅝ', 'ㅙ'],
    'ㅟ': ['ㅚ', 'ㅜ'],
    'ㅠ': ['ㅜ', 'ㅛ'],
    'ㅡ': ['ㅣ', 'ㅜ'],
    'ㅢ': ['ㅣ', 'ㅡ'],
    'ㅣ': ['ㅡ', 'ㅢ']
}

# 키보드 인접 자모 매핑 (두벌식 기준)
KEYBOARD_ADJACENT_CHOSUNG = {
    'ㅂ': ['ㅈ', 'ㅁ'],
    'ㅈ': ['ㅂ', 'ㄷ', 'ㄴ'],
    'ㄷ': ['ㅈ', 'ㄱ', 'ㅇ'],
    'ㄱ': ['ㄷ', 'ㅅ', 'ㄹ'],
    'ㅅ': ['ㄱ', 'ㅇ', 'ㅎ'],
    'ㅁ': ['ㅂ', 'ㄴ'],
    'ㄴ': ['ㅈ', 'ㅁ', 'ㅇ'],
    'ㅇ': ['ㄷ', 'ㄴ', 'ㅅ', 'ㄹ'],
    'ㄹ': ['ㄱ', 'ㅇ', 'ㅎ'],
    'ㅎ': ['ㅅ', 'ㄹ'],
    'ㅋ': ['ㅌ'],
    'ㅌ': ['ㅋ', 'ㅊ'],
    'ㅊ': ['ㅌ', 'ㅍ'],
    'ㅍ': ['ㅊ']
}

KEYBOARD_ADJACENT_JUNGSUNG = {
    'ㅛ': ['ㅕ', 'ㅗ'],
    'ㅕ': ['ㅛ', 'ㅓ', 'ㅗ', 'ㅏ'],
    'ㅗ': ['ㅛ', 'ㅕ', 'ㅓ', 'ㅏ', 'ㅜ'],
    'ㅓ': ['ㅕ', 'ㅗ', 'ㅏ', 'ㅜ', 'ㅡ'],
    'ㅏ': ['ㅕ', 'ㅗ', 'ㅓ', 'ㅜ', 'ㅡ', 'ㅣ'],
    'ㅜ': ['ㅗ', 'ㅓ', 'ㅏ', 'ㅡ', 'ㅣ'],
    'ㅡ': ['ㅓ', 'ㅏ', 'ㅜ', 'ㅣ'],
    'ㅣ': ['ㅏ', 'ㅜ', 'ㅡ']
}

def decompose_hangul(char: str) -> Tuple[int, int, int]:
    """한글 문자를 초성, 중성, 종성으로 분해"""
    if '가' <= char <= '힣':
        code = ord(char) - HANGUL_BASE
        cho = code // 588
        jung = (code % 588) // 28
        jong = code % 28
        return cho, jung, jong
    return -1, -1, -1

def compose_hangul(cho: int, jung: int, jong: int) -> str:
    """초성, 중성, 종성을 한글 문자로 조합"""
    if 0 <= cho < 19 and 0 <= jung < 21 and 0 <= jong < 28:
        code = HANGUL_BASE + (cho * 588) + (jung * 28) + jong
        return chr(code)
    return ''

def is_hangul(char: str) -> bool:
    """한글 음절인지 확인"""
    return '가' <= char <= '힣'

def get_hangul_positions(text: str) -> List[int]:
    """문장에서 한글 문자의 위치를 반환"""
    return [i for i, char in enumerate(text) if is_hangul(char)]

def get_context_string(text: str, pos: int, window: int = 1) -> str:
    """위치 주변의 문자열 컨텍스트를 가져옴"""
    start = max(0, pos - window)
    end = min(len(text), pos + window + 1)
    context = text[start:pos] + text[pos:end]
    return context

# 1. 교체 (Substitution) 함수들
def apply_substitution(text: str, num_errors: int = 1, used_positions: Set[int] = None, errors_list: List[str] = None) -> Tuple[str, Set[int], List[str]]:
    """교체 오타를 적용"""
    text_list = list(text)
    hangul_positions = get_hangul_positions(text)
    
    if errors_list is None:
        errors_list = []
    
    if not hangul_positions:
        return text, set(), errors_list
    
    if used_positions:
        available_positions = [pos for pos in hangul_positions if pos not in used_positions]
    else:
        available_positions = hangul_positions
        used_positions = set()
    
    if not available_positions:
        return text, used_positions, errors_list
    
    # 실제로 교체된 개수를 추적
    errors_applied = 0
    attempts = 0
    max_attempts = len(available_positions) * 5  # 충분한 재시도 횟수
    
    while errors_applied < num_errors and available_positions and attempts < max_attempts:
        pos = random.choice(available_positions)
        original_char = text_list[pos]
        
        # 모든 교체 방법을 시도
        substitution_types = ['similar_jamo', 'keyboard_adjacent', 'phonetic']
        random.shuffle(substitution_types)
        
        new_char = None
        for sub_type in substitution_types:
            if sub_type == 'similar_jamo':
                temp_char = substitute_similar_jamo(original_char)
            elif sub_type == 'keyboard_adjacent':
                temp_char = substitute_keyboard_adjacent(original_char)
            else:  # phonetic
                temp_char = substitute_phonetic(original_char)
            
            if temp_char != original_char:
                new_char = temp_char
                break
        
        # 모든 방법이 실패하면 강제로 랜덤 교체
        if new_char is None or new_char == original_char:
            new_char = substitute_force_random(original_char)
        
        if new_char != original_char:
            text_list[pos] = new_char
            errors_list.append(f"{original_char} -> {new_char}")
            errors_applied += 1
        
        # 성공 여부와 관계없이 해당 위치는 사용된 것으로 표시
        used_positions.add(pos)
        available_positions.remove(pos)
        attempts += 1
    
    return ''.join(text_list), used_positions, errors_list

def substitute_force_random(char: str) -> str:
    """강제로 랜덤 자모로 교체 (반드시 다른 글자로 변경)"""
    cho, jung, jong = decompose_hangul(char)
    if cho == -1:
        return char
    
    # 초성, 중성, 종성 중 하나를 랜덤하게 변경
    change_type = random.choice(['cho', 'jung', 'jong'])
    
    if change_type == 'cho':
        # 현재와 다른 초성으로 변경
        new_cho = random.choice([i for i in range(19) if i != cho])
        return compose_hangul(new_cho, jung, jong)
    elif change_type == 'jung':
        # 현재와 다른 중성으로 변경
        new_jung = random.choice([i for i in range(21) if i != jung])
        return compose_hangul(cho, new_jung, jong)
    else:
        # 종성 변경 (없으면 추가, 있으면 제거하거나 변경)
        if jong == 0:
            new_jong = random.choice(range(1, 28))
        else:
            new_jong = random.choice([i for i in range(28) if i != jong])
        return compose_hangul(cho, jung, new_jong)

def substitute_similar_jamo(char: str) -> str:
    """유사한 자모로 교체 - 더 적극적으로 교체"""
    cho, jung, jong = decompose_hangul(char)
    if cho == -1:
        return char
    
    # 초성과 중성 모두 교체 시도
    cho_char = CHOSUNG_LIST[cho]
    jung_char = JUNGSUNG_LIST[jung]
    
    # 초성 교체 가능 여부 확인
    can_replace_cho = cho_char in SIMILAR_CHOSUNG
    # 중성 교체 가능 여부 확인
    can_replace_jung = jung_char in SIMILAR_JUNGSUNG
    
    if can_replace_cho and can_replace_jung:
        # 둘 다 가능하면 랜덤 선택
        if random.random() < 0.5:
            new_cho_char = random.choice(SIMILAR_CHOSUNG[cho_char])
            new_cho = CHOSUNG_LIST.index(new_cho_char)
            return compose_hangul(new_cho, jung, jong)
        else:
            new_jung_char = random.choice(SIMILAR_JUNGSUNG[jung_char])
            new_jung = JUNGSUNG_LIST.index(new_jung_char)
            return compose_hangul(cho, new_jung, jong)
    elif can_replace_cho:
        # 초성만 교체 가능
        new_cho_char = random.choice(SIMILAR_CHOSUNG[cho_char])
        new_cho = CHOSUNG_LIST.index(new_cho_char)
        return compose_hangul(new_cho, jung, jong)
    elif can_replace_jung:
        # 중성만 교체 가능
        new_jung_char = random.choice(SIMILAR_JUNGSUNG[jung_char])
        new_jung = JUNGSUNG_LIST.index(new_jung_char)
        return compose_hangul(cho, new_jung, jong)
    
    return char

def substitute_keyboard_adjacent(char: str) -> str:
    """키보드 인접 자모로 교체 - 더 적극적으로 교체"""
    cho, jung, jong = decompose_hangul(char)
    if cho == -1:
        return char
    
    cho_char = CHOSUNG_LIST[cho]
    jung_char = JUNGSUNG_LIST[jung]
    
    # 교체 가능 여부 확인
    can_replace_cho = cho_char in KEYBOARD_ADJACENT_CHOSUNG
    can_replace_jung = jung_char in KEYBOARD_ADJACENT_JUNGSUNG
    
    if can_replace_cho and can_replace_jung:
        # 둘 다 가능하면 랜덤 선택
        if random.random() < 0.5:
            new_cho_char = random.choice(KEYBOARD_ADJACENT_CHOSUNG[cho_char])
            new_cho = CHOSUNG_LIST.index(new_cho_char)
            return compose_hangul(new_cho, jung, jong)
        else:
            new_jung_char = random.choice(KEYBOARD_ADJACENT_JUNGSUNG[jung_char])
            new_jung = JUNGSUNG_LIST.index(new_jung_char)
            return compose_hangul(cho, new_jung, jong)
    elif can_replace_cho:
        # 초성만 교체 가능
        new_cho_char = random.choice(KEYBOARD_ADJACENT_CHOSUNG[cho_char])
        new_cho = CHOSUNG_LIST.index(new_cho_char)
        return compose_hangul(new_cho, jung, jong)
    elif can_replace_jung:
        # 중성만 교체 가능
        new_jung_char = random.choice(KEYBOARD_ADJACENT_JUNGSUNG[jung_char])
        new_jung = JUNGSUNG_LIST.index(new_jung_char)
        return compose_hangul(cho, new_jung, jong)
    
    return char

def substitute_random_jamo(char: str) -> str:
    """랜덤 자모로 교체 (매핑에 없는 경우를 위한 백업)"""
    cho, jung, jong = decompose_hangul(char)
    if cho == -1:
        return char
    
    # 랜덤하게 초성 또는 중성 교체
    if random.random() < 0.5:
        # 초성 교체 - 현재와 다른 랜덤 초성 선택
        new_cho = random.choice([i for i in range(19) if i != cho])
        return compose_hangul(new_cho, jung, jong)
    else:
        # 중성 교체 - 현재와 다른 랜덤 중성 선택
        new_jung = random.choice([i for i in range(21) if i != jung])
        return compose_hangul(cho, new_jung, jong)
    
    return char

def substitute_phonetic(char: str) -> str:
    """음운적으로 유사한 문자로 교체"""
    # 간단한 예시 구현
    phonetic_map = {
        '지': '치', '치': '지',
        '자': '차', '차': '자',
        '즈': '츠', '츠': '즈'
    }
    
    if char in phonetic_map:
        return phonetic_map[char]
    
    # 기본적으로 similar_jamo로 대체
    return substitute_similar_jamo(char)

# 2. 삭제 (Deletion) 함수들
def apply_deletion(text: str, num_errors: int = 1, used_positions: Set[int] = None, errors_list: List[str] = None) -> Tuple[str, Set[int], List[str]]:
    """삭제 오타를 적용"""
    text_list = list(text)
    original_text = ''.join(text_list)
    if used_positions is None:
        used_positions = set()
    if errors_list is None:
        errors_list = []
    
    for _ in range(num_errors):
        if random.random() < 0.5:
            # 자모 삭제
            result = delete_jamo(text_list, used_positions)
            if result:
                text_list, used_positions, error_desc = result
                if error_desc:
                    errors_list.append(error_desc)
        else:
            # 음절 삭제
            result = delete_syllable(text_list, used_positions)
            if result:
                text_list, used_positions, error_desc = result
                if error_desc:
                    errors_list.append(error_desc)
    
    return ''.join(text_list), used_positions, errors_list

def delete_jamo(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """자모를 삭제"""
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    if not hangul_positions:
        return None
    
    pos = random.choice(hangul_positions)
    char = text_list[pos]
    cho, jung, jong = decompose_hangul(char)
    
    original_char = char
    error_desc = ""
    
    if jong != 0:  # 종성이 있으면 종성 삭제
        new_char = compose_hangul(cho, jung, 0)
        text_list[pos] = new_char
        error_desc = f"{original_char} -> {new_char}"
        used_positions.add(pos)
    else:
        # 초성 삭제 (분리된 형태로)
        new_char = JUNGSUNG_LIST[jung]
        text_list[pos] = new_char
        error_desc = f"{original_char} -> {new_char}"
        used_positions.add(pos)
    
    return text_list, used_positions, error_desc

def delete_syllable(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """음절을 삭제"""
    if len(text_list) <= 1:
        return None
    
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    if not hangul_positions:
        return None
    
    pos = random.choice(hangul_positions)
    deleted_char = text_list[pos]
    
    # 삭제할 때는 위치를 추적하기 어려우므로 삭제된 위치 이후 위치를 조정
    del text_list[pos]
    
    # 삭제 후 인덱스 조정
    new_used = set()
    for p in used_positions:
        if p < pos:
            new_used.add(p)
        elif p > pos:
            new_used.add(p - 1)
    
    error_desc = f"{deleted_char} -> (삭제됨)"
    
    return text_list, new_used, error_desc

# 3. 추가 (Insertion) 함수들
def apply_insertion(text: str, num_errors: int = 1, used_positions: Set[int] = None, errors_list: List[str] = None) -> Tuple[str, Set[int], List[str]]:
    """추가 오타를 적용"""
    text_list = list(text)
    if used_positions is None:
        used_positions = set()
    if errors_list is None:
        errors_list = []
    
    for _ in range(num_errors):
        result = insert_jamo(text_list, used_positions)
        if result:
            text_list, used_positions, error_desc = result
            if error_desc:
                errors_list.append(error_desc)

    
    return ''.join(text_list), used_positions, errors_list

def insert_jamo(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """자모를 추가"""
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    if not hangul_positions:
        return None
    
    pos = random.choice(hangul_positions)
    char = text_list[pos]
    cho, jung, jong = decompose_hangul(char)
    
    original_char = char
    error_desc = ""
    
    if jong == 0:  # 종성이 없으면 종성 추가
        new_jong = random.choice(range(1, 28))
        new_char = compose_hangul(cho, jung, new_jong)
        text_list[pos] = new_char
        error_desc = f"{original_char} -> {new_char}"
        used_positions.add(pos)
    else:  # 종성이 있으면 분리된 형태로 추가
        text_list[pos] = char + JONGSUNG_LIST[jong]
        error_desc = f"{original_char} -> {char}{JONGSUNG_LIST[jong]}"
        used_positions.add(pos)
    
    return text_list, used_positions, error_desc

def insert_syllable(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """음절을 추가 (중복)"""
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    if not hangul_positions:
        return None
    
    pos = random.choice(hangul_positions)
    char_to_duplicate = text_list[pos]
    
    # 해당 위치의 문자를 중복
    text_list.insert(pos, char_to_duplicate)
    
    # 삽입 후 인덱스 조정
    new_used = set()
    for p in used_positions:
        if p >= pos:
            new_used.add(p + 1)
        else:
            new_used.add(p)
    new_used.add(pos)  # 삽입된 위치도 사용된 것으로 표시
    
    error_desc = f"{char_to_duplicate} -> {char_to_duplicate}{char_to_duplicate}"
    
    return text_list, new_used, error_desc

# 4. 전치 (Transposition) 함수들
def apply_transposition(text: str, num_errors: int = 1, used_positions: Set[int] = None, errors_list: List[str] = None) -> Tuple[str, Set[int], List[str]]:
    """전치 오타를 적용"""
    text_list = list(text)
    if used_positions is None:
        used_positions = set()
    if errors_list is None:
        errors_list = []
    
    for _ in range(num_errors):
        # if random.random() < 0.5:
        #     result = transpose_jamo(text_list, used_positions)
        #     if result:
        #         text_list, used_positions, error_desc = result
        #         if error_desc:
        #             errors_list.append(error_desc)
        # else:
        #     # 음절 전치
        #     result = transpose_syllable(text_list, used_positions)
        #     if result:
        #         text_list, used_positions, error_desc = result
        #         if error_desc:
        #             errors_list.append(error_desc)
       
        result = transpose_jamo(text_list, used_positions)
        if result:
            text_list, used_positions, error_desc = result
            if error_desc:
                errors_list.append(error_desc)

        
    return ''.join(text_list), used_positions, errors_list

def transpose_jamo(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """자모 순서를 전치 (분리된 형태로 - 예: 호 → ㅗㅎ)"""
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    if not hangul_positions:
        return None
    
    pos = random.choice(hangul_positions)
    char = text_list[pos]
    cho, jung, jong = decompose_hangul(char)
    
    original_char = char
    
    # 자모를 분리된 형태로 표현
    if jong != 0:
        # 종성이 있으면 초성+중성+종성 중에서 순서 바꾸기
        if random.random() < 0.5:
            # 중성을 앞으로
            new_char = JUNGSUNG_LIST[jung] + CHOSUNG_LIST[cho] + JONGSUNG_LIST[jong]
        else:
            # 초성과 중성 위치 교환, 종성은 그대로
            new_char = JUNGSUNG_LIST[jung] + CHOSUNG_LIST[cho]
            if jong != 0:
                new_char += JONGSUNG_LIST[jong]
    else:
        # 종성이 없으면 초성과 중성만 교환 (예: 호 → ㅗㅎ)
        new_char = JUNGSUNG_LIST[jung] + CHOSUNG_LIST[cho]
    
    text_list[pos] = new_char
    used_positions.add(pos)
    
    error_desc = f"{original_char} -> {new_char}"
    
    return text_list, used_positions, error_desc

def transpose_syllable(text_list: List[str], used_positions: Set[int]) -> Optional[Tuple[List[str], Set[int], str]]:
    """인접한 음절 순서를 전치"""
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char)]
    
    # 사용되지 않은 연속된 한글 위치 찾기
    for i in range(len(hangul_positions) - 1):
        pos1, pos2 = hangul_positions[i], hangul_positions[i+1]
        if pos2 - pos1 == 1 and pos1 not in used_positions and pos2 not in used_positions:
            char1, char2 = text_list[pos1], text_list[pos2]
            text_list[pos1], text_list[pos2] = text_list[pos2], text_list[pos1]
            used_positions.add(pos1)
            used_positions.add(pos2)
            
            error_desc = f"{char1}{char2} -> {char2}{char1}"
            return text_list, used_positions, error_desc
    
    return None

# 5. 띄어쓰기 오류 (Spacing) 함수들
def apply_spacing_error(text: str, num_errors: int = 1, used_positions: Set[int] = None, errors_list: List[str] = None) -> Tuple[str, Set[int], List[str]]:
    """띄어쓰기 오타를 적용 (자모 분리 포함)"""
    if used_positions is None:
        used_positions = set()
    if errors_list is None:
        errors_list = []
    
    for _ in range(num_errors):
        error_type = random.choice(['remove_space', 'add_space_between_syllables', 'add_space_in_jamo'])
        
        if error_type == 'remove_space' and ' ' in text:
            # 공백 삭제
            result = remove_space(text, used_positions)
            if result:
                text, new_pos, error_desc = result
                if error_desc:
                    errors_list.append(error_desc)
                    used_positions.update(new_pos)
        elif error_type == 'add_space_between_syllables':
            # 음절 사이에 공백 추가
            result = add_space(text, used_positions)
            if result:
                text, new_pos, error_desc = result
                if error_desc:
                    errors_list.append(error_desc)
                    used_positions.update(new_pos)
        else:  # add_space_in_jamo
            # 자모 사이에 공백 추가 (예: 국 → ㄱ ㅜㄱ)
            result = add_space_in_jamo(text, used_positions)
            if result:
                text, new_pos, error_desc = result
                if error_desc:
                    errors_list.append(error_desc)
                    used_positions.update(new_pos)
    
    return text, used_positions, errors_list
def add_space_in_jamo(text: str, used_positions: Set[int]) -> Optional[Tuple[str, Set[int], str]]:
    """자모 사이에 공백을 추가 (예: 국 → 구 ㄱ 또는 ㄱ ㅜㄱ)"""
    text_list = list(text)
    hangul_positions = [i for i, char in enumerate(text_list) if is_hangul(char) and i not in used_positions]
    
    if not hangul_positions:
        return None
    
    # 랜덤하게 한글 글자 선택
    pos = random.choice(hangul_positions)
    char = text_list[pos]
    cho, jung, jong = decompose_hangul(char)
    
    if cho == -1:
        return None
    
    original_char = char
    
    if jong != 0:  # 종성이 있는 경우
        if random.random() < 0.5:
            # 종성을 분리 (예: 국 → 구 ㄱ)
            # 초성+중성을 다시 조합하고, 종성은 별도로
            new_char = compose_hangul(cho, jung, 0)  # 종성 없는 글자로 재조합
            if new_char:
                jamo_str = new_char + ' ' + JONGSUNG_LIST[jong]
            else:
                # 조합 실패 시 초성과 중성+종성 사이에 공백
                jamo_str = CHOSUNG_LIST[cho] + ' ' + JUNGSUNG_LIST[jung] + JONGSUNG_LIST[jong]
        else:
            # 초성과 중성+종성 사이에 공백 (예: 국 → ㄱ ㅜㄱ)
            jamo_str = CHOSUNG_LIST[cho] + ' ' + JUNGSUNG_LIST[jung] + JONGSUNG_LIST[jong]
    else:  # 종성이 없는 경우
        # 초성과 중성 사이에 공백 (예: 가 → ㄱ ㅏ)
        jamo_str = CHOSUNG_LIST[cho] + ' ' + JUNGSUNG_LIST[jung]
    
    # 원래 글자를 분리된 자모로 교체
    text_list[pos] = jamo_str
    
    error_desc = f"{original_char} -> {jamo_str}"
    
    return ''.join(text_list), {pos}, error_desc

def remove_space(text: str, used_positions: Set[int]) -> Optional[Tuple[str, Set[int], str]]:
    """공백을 제거"""
    spaces = [i for i, char in enumerate(text) if char == ' ' and i not in used_positions]
    if not spaces:
        return None
    
    pos = random.choice(spaces)
    
    # 공백 주변의 컨텍스트 가져오기
    before = text[max(0, pos-1):pos] if pos > 0 else ""
    after = text[pos+1:min(len(text), pos+2)] if pos < len(text)-1 else ""
    
    original_context = f"{before} {after}"
    new_context = f"{before}{after}"
    
    text = text[:pos] + text[pos+1:]
    
    error_desc = f"{original_context} -> {new_context}"
    
    return text, {pos}, error_desc

def add_space(text: str, used_positions: Set[int]) -> Optional[Tuple[str, Set[int], str]]:
    """불필요한 공백을 추가 (음절 사이)"""
    # 조사 앞에 공백 추가
    particles = ['을', '를', '이', '가', '은', '는', '와', '과', '에', '에서', '으로', '로', '의']
    
    for particle in particles:
        if particle in text:
            # 조사의 위치 찾기
            idx = text.find(particle)
            while idx != -1:
                if idx > 0 and idx not in used_positions and text[idx-1] != ' ':
                    before = text[max(0, idx-2):idx]
                    original_context = f"{before}{particle}"
                    new_context = f"{before} {particle}"
                    
                    text = text[:idx] + ' ' + text[idx:]
                    error_desc = f"{original_context} -> {new_context}"
                    return text, {idx}, error_desc
                idx = text.find(particle, idx + 1)
    
    # 랜덤 위치에 공백 추가
    hangul_positions = get_hangul_positions(text)
    available = [i for i in hangul_positions[1:] if i not in used_positions]
    if available:
        pos = random.choice(available)
        before = text[max(0, pos-1):pos]
        after = text[pos:min(len(text), pos+1)]
        
        original_context = f"{before}{after}"
        new_context = f"{before} {after}"
        
        text = text[:pos] + ' ' + text[pos:]
        error_desc = f"{original_context} -> {new_context}"
        return text, {pos}, error_desc
    
    return None

def generate_typos_for_sentence(sentence: str, sentence_id: int = None) -> Dict:
    """문장에 대해 모든 타입의 오타를 생성"""
    result = {"original": sentence}

    # ID 추가
    if sentence_id is not None:
        result["id"] = sentence_id
    
    # 1. Substitution
    sub_1, used_pos_sub, errors_1 = apply_substitution(sentence, 1)
    sub_2, _, errors_2 = apply_substitution(sub_1, 1, used_pos_sub, errors_1.copy())
    result["substitution"] = {
        "1_error": {"text": sub_1, "errors": errors_1},
        "2_errors": {"text": sub_2, "errors": errors_2}
    }
    
    # 2. Deletion
    del_1, used_pos_del, errors_1 = apply_deletion(sentence, 1)
    del_2, _, errors_2 = apply_deletion(del_1, 1, used_pos_del, errors_1.copy())
    result["deletion"] = {
        "1_error": {"text": del_1, "errors": errors_1},
        "2_errors": {"text": del_2, "errors": errors_2}
    }
    
    # 3. Insertion
    ins_1, used_pos_ins, errors_1 = apply_insertion(sentence, 1)
    ins_2, _, errors_2 = apply_insertion(ins_1, 1, used_pos_ins, errors_1.copy())
    result["insertion"] = {
        "1_error": {"text": ins_1, "errors": errors_1},
        "2_errors": {"text": ins_2, "errors": errors_2}
    }
    
    # 4. Transposition
    trans_1, used_pos_trans, errors_1 = apply_transposition(sentence, 1)
    trans_2, _, errors_2 = apply_transposition(trans_1, 1, used_pos_trans, errors_1.copy())
    result["transposition"] = {
        "1_error": {"text": trans_1, "errors": errors_1},
        "2_errors": {"text": trans_2, "errors": errors_2}
    }
    
    # 5. Spacing
    space_1, used_pos_space, errors_1 = apply_spacing_error(sentence, 1)
    space_2, _, errors_2 = apply_spacing_error(space_1, 1, used_pos_space, errors_1.copy())
    result["spacing"] = {
        "1_error": {"text": space_1, "errors": errors_1},
        "2_errors": {"text": space_2, "errors": errors_2}
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Generate Korean typos from input JSON file')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # 입력 파일 읽기
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 입력이 문자열 리스트인지 확인
    if not isinstance(input_data, list):
        raise ValueError("Input must be a list of strings")
    
    # 각 문장에 대해 오타 생성
    results = []
    for idx, sentence in enumerate(input_data, 1):  # 1부터 시작하는 ID
        if isinstance(sentence, str):
            result = generate_typos_for_sentence(sentence, sentence_id=idx)
            results.append(result)
    
    # 결과를 JSON 파일로 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Typo generation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()