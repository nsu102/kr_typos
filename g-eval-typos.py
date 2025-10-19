"""
G-eval을 사용한 오탈자 데이터 평가 시스템
오탈자 1개 vs 1개, 2개 vs 2개를 비교하여 어떤 응답이 더 좋은지 평가
"""

import json
import os
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
from tqdm import tqdm

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

@dataclass
class ComparisonResult:
    """비교 결과를 담는 데이터 클래스"""
    original_question: str
    error_type: str  # "1_error" or "2_errors"
    model_a: str
    model_b: str
    question_a: str
    question_b: str
    response_a: str
    response_b: str
    winner: str  # "A" or "B"
    score_a: float
    score_b: float
    metrics_a: Dict  # 개별 메트릭 점수
    metrics_b: Dict  # 개별 메트릭 점수


class GEvalTyposEvaluator:
    """G-eval을 사용한 오탈자 응답 평가기"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.g_eval_prompt = self._create_g_eval_prompt()

    def _create_g_eval_prompt(self) -> str:
            return """You will be given one original question (without typos) and two versions of the same question with typos, along with their corresponding responses from two different models.

    Your task is to rate the quality of each response based on how well the model handled the typos in the question.

    Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

    Evaluation Criteria:

    Helpfulness (1-10) - Evaluate how effectively the response meets the user's requirements and whether it provides practical and insightful information.

    Relevance (1-10) - Measure whether the response is directly connected to the question or topic, and evaluate whether it focuses on the query context without unnecessary information.

    Accuracy (1-10) - Assess whether the information provided is factually accurate and reliable, and verify that it avoids errors, misinformation, or ambiguity.

    Depth (1-10) - Evaluate the level of detail or comprehensiveness of the response, and examine whether it goes beyond the surface of the topic to provide clear explanations, meaningful context, and thoughtful analysis.

    Evaluation Steps:

    1. Read the original question (without typos) carefully to understand the true intent.
    2. Read each question with typos and identify what errors exist.
    3. Read Response A and evaluate how helpful it is in meeting the user's needs despite typos.
    4. Read Response B and evaluate how helpful it is in meeting the user's needs despite typos.
    5. For each response, assess whether it stays relevant to the question without unnecessary information.
    6. For each response, verify the factual accuracy and reliability of the information provided.
    7. For each response, check the depth - whether it provides detailed explanations, meaningful context, and thoughtful analysis.
    8. Assign scores for each metric (Helpfulness, Relevance, Accuracy, Depth) on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the Evaluation Criteria.
    9. Determine the winner based on the overall quality. NO TIES ALLOWED - you must select a clear winner (A or B).

    Example:

    Original Question (No Typos):
    {original_question}

    Question A (with typos):
    {question_a}
    Typos: {errors_a}
    Model A: {model_a}

    Response A:
    {response_a}

    Question B (with typos):
    {question_b}
    Typos: {errors_b}
    Model B: {model_b}

    Response B:
    {response_b}

    Evaluation Form (scores ONLY):

    {{
        "helpfulness_a": <score 1-10>,
        "relevance_a": <score 1-10>,
        "accuracy_a": <score 1-10>,
        "depth_a": <score 1-10>,
        "helpfulness_b": <score 1-10>,
        "relevance_b": <score 1-10>,
        "accuracy_b": <score 1-10>,
        "depth_b": <score 1-10>,
        "winner": "<A or B, NO ties allowed>"
    }}
    """

    def evaluate_pair(
        self,
        original_question: str,
        question_a: str,
        errors_a: List[str],
        model_a: str,
        response_a: str,
        question_b: str,
        errors_b: List[str],
        model_b: str,
        response_b: str
    ) -> Dict:
        """두 응답을 G-eval로 비교 평가 (논문 방식)"""

        prompt = self.g_eval_prompt.format(
            original_question=original_question,
            question_a=question_a,
            errors_a=", ".join(errors_a),
            model_a=model_a,
            response_a=response_a[:2000],
            question_b=question_b,
            errors_b=", ".join(errors_b),
            model_b=model_b,
            response_b=response_b[:2000]
        )

        
        max_retries = 5
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator for assessing the quality of language model responses. You must respond with a valid JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # 논문에서 사용한 temperature=0
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content.strip()

                result = json.loads(result_text)
                break 

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    print(error_str)
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⏳ Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Max retries reached for rate limit error")
                        raise
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        try:

            # 4개 메트릭의 평균으로 최종 점수 계산 (1-10 척도)
            score_a = (
                result.get("helpfulness_a", 5) +
                result.get("relevance_a", 5) +
                result.get("accuracy_a", 5) +
                result.get("depth_a", 5)
            ) / 4.0

            score_b = (
                result.get("helpfulness_b", 5) +
                result.get("relevance_b", 5) +
                result.get("accuracy_b", 5) +
                result.get("depth_b", 5)
            ) / 4.0

            # 승자는 JSON에서 직접 가져오기 (NO ties)
            winner = result.get("winner", "A").upper()

            # 안전장치: winner가 A나 B가 아닌 경우 점수로 판단
            if winner not in ["A", "B"]:
                winner = "A" if score_a >= score_b else "B"

            return {
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "metrics_a": {
                    "helpfulness": result.get("helpfulness_a", 5),
                    "relevance": result.get("relevance_a", 5),
                    "accuracy": result.get("accuracy_a", 5),
                    "depth": result.get("depth_a", 5)
                },
                "metrics_b": {
                    "helpfulness": result.get("helpfulness_b", 5),
                    "relevance": result.get("relevance_b", 5),
                    "accuracy": result.get("accuracy_b", 5),
                    "depth": result.get("depth_b", 5)
                }
            }

        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return {
                "score_a": 5.0,
                "score_b": 5.0,
                "winner": "A",  # 오류 시 기본값
                "metrics_a": {"helpfulness": 5, "relevance": 5, "accuracy": 5, "depth": 5},
                "metrics_b": {"helpfulness": 5, "relevance": 5, "accuracy": 5, "depth": 5}
            }

    def load_data(self, file_path: str) -> List[Dict]:
        """JSON 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_comparison_pairs(self, data: List[Dict]) -> List[Tuple]:
        """비교 쌍 생성: 5C2 방식으로 같은 모델 내에서 타입 간 비교"""
        from itertools import combinations
        
        comparison_pairs = []
        TYPO_TYPES = ['substitution', 'deletion', 'insertion', 'transposition', 'spacing']
        ERROR_LEVELS = ['1_error', '2_errors']
        MODELS = ['qwen_72b', 'qwen_7b']

        for item in data:
            original = item['original']
            
            # 각 error level과 model에 대해
            for error_level in ERROR_LEVELS:
                for model in MODELS:
                    # 5가지 타입 중 2개를 선택 (5C2 = 10)
                    for type_a, type_b in combinations(TYPO_TYPES, 2):
                        # type_a 데이터 확인
                        if type_a not in item:
                            continue
                        if error_level not in item[type_a]:
                            continue
                        entry_a = item[type_a][error_level]
                        if 'responses' not in entry_a or model not in entry_a['responses']:
                            continue
                        
                        # type_b 데이터 확인
                        if type_b not in item:
                            continue
                        if error_level not in item[type_b]:
                            continue
                        entry_b = item[type_b][error_level]
                        if 'responses' not in entry_b or model not in entry_b['responses']:
                            continue
                        
                        # 비교 쌍 생성
                        comparison_pairs.append({
                            'type': f'{error_level}_{model}_{type_a}_vs_{type_b}',
                            'original': original,
                            'question_a': entry_a['text'],
                            'errors_a': entry_a.get('errors', []),
                            'question_b': entry_b['text'],
                            'errors_b': entry_b.get('errors', []),
                            'model_a': model,
                            'model_b': model,
                            'response_a': entry_a['responses'][model],
                            'response_b': entry_b['responses'][model]
                        })

        return comparison_pairs

    def process_single_comparison(self, pair: Dict) -> ComparisonResult:
        """단일 비교 쌍 처리"""
        eval_result = self.evaluate_pair(
            original_question=pair['original'],
            question_a=pair['question_a'],
            errors_a=pair['errors_a'],
            model_a=pair['model_a'],
            response_a=pair['response_a'],
            question_b=pair['question_b'],
            errors_b=pair['errors_b'],
            model_b=pair['model_b'],
            response_b=pair['response_b']
        )

        return ComparisonResult(
            original_question=pair['original'],
            error_type=pair['type'],
            model_a=pair['model_a'],
            model_b=pair['model_b'],
            question_a=pair['question_a'],
            question_b=pair['question_b'],
            response_a=pair['response_a'],
            response_b=pair['response_b'],
            winner=eval_result['winner'],
            score_a=eval_result['score_a'],
            score_b=eval_result['score_b'],
            metrics_a=eval_result['metrics_a'],
            metrics_b=eval_result['metrics_b']
        )

    def run_evaluation(
        self,
        data_file: str,
        output_file: str = "data/outputs/typos/g_eval_results.json",
        max_workers: int = 30,
        limit: int = None,
        checkpoint_every: int = 50
    ):
        """멀티스레딩으로 평가 실행 (checkpoint 지원)"""

        print(f"📊 데이터 로딩 중: {data_file}")
        data = self.load_data(data_file)

        print(f"🔄 비교 쌍 생성 중...")
        comparison_pairs = self.create_comparison_pairs(data)

        if limit:
            comparison_pairs = comparison_pairs[:limit]

        print(f"✅ 총 {len(comparison_pairs)}개의 비교 쌍 생성됨")
        
        # 체크포인트 파일 확인 및 로드
        checkpoint_file = output_file + ".ckpt.json"
        results = []
        completed_indices = set()
        
        if os.path.exists(checkpoint_file):
            print(f"📥 체크포인트 발견: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    results = [ComparisonResult(**item) for item in checkpoint_data]
                    completed_indices = set(range(len(results)))
                print(f"✅ 기존 {len(results)}개 결과 로드됨")
            except Exception as e:
                print(f"⚠️  체크포인트 로드 실패: {e}")
                results = []
                completed_indices = set()
        
        # 남은 작업만 필터링
        remaining_pairs = [
            (idx, pair) for idx, pair in enumerate(comparison_pairs)
            if idx not in completed_indices
        ]
        
        if not remaining_pairs:
            print("✅ 모든 평가가 이미 완료되었습니다!")
            self._print_statistics(results)
            return results
        
        print(f"🚀 {max_workers}개의 스레드로 평가 시작... (남은 작업: {len(remaining_pairs)}개)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_comparison, pair): (idx, pair)
                for idx, pair in remaining_pairs
            }

            with tqdm(total=len(remaining_pairs), desc="평가 진행") as pbar:
                completed_count = 0
                for future in as_completed(futures):
                    idx, pair = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                        pbar.update(1)
                        
                        # Checkpoint 저장
                        if completed_count % checkpoint_every == 0:
                            self._save_checkpoint(results, checkpoint_file)
                            print(f"💾 체크포인트 저장: {completed_count}개 완료")
                            
                    except Exception as e:
                        print(f"❌ 오류 발생 (idx={idx}): {e}")
                        pbar.update(1)

        self._save_results(results, output_file)        
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return results
    
    def _save_checkpoint(self, results: List[ComparisonResult], checkpoint_file: str):
        """체크포인트 저장"""
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        results_dict = [
            {
                'original_question': r.original_question,
                'error_type': r.error_type,
                'model_a': r.model_a,
                'model_b': r.model_b,
                'question_a': r.question_a,
                'question_b': r.question_b,
                'response_a': r.response_a,
                'response_b': r.response_b,
                'winner': r.winner,
                'score_a': r.score_a,
                'score_b': r.score_b,
                'metrics_a': r.metrics_a,
                'metrics_b': r.metrics_b
            }
            for r in results
        ]
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    def _save_results(self, results: List[ComparisonResult], output_file: str):
        """결과를 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        results_dict = [
            {
                'original_question': r.original_question,
                'error_type': r.error_type,
                'model_a': r.model_a,
                'model_b': r.model_b,
                'question_a': r.question_a,
                'question_b': r.question_b,
                'response_a': r.response_a,
                'response_b': r.response_b,
                'winner': r.winner,
                'score_a': r.score_a,
                'score_b': r.score_b,
                'metrics_a': r.metrics_a,
                'metrics_b': r.metrics_b
            }
            for r in results
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="G-Eval을 사용한 오탈자 응답 평가")
    parser.add_argument("--input_file",
                        default="data/outputs/typos/typos_data_filled.json",
                        help="입력 데이터 파일")
    parser.add_argument("--output_file",
                        default="data/outputs/typos/g_eval_results.json",
                        help="출력 결과 파일")
    parser.add_argument("--model",
                        default="openai/gpt-4o-mini",
                        help="평가에 사용할 모델 (OpenRouter format: openai/gpt-4o-mini)")
    parser.add_argument("--workers",
                        type=int,
                        default=3,
                        help="멀티스레딩 워커 수 (rate limit 회피를 위해 낮게 설정 권장)")
    parser.add_argument("--limit",
                        type=int,
                        default=None,
                        help="평가할 항목 수 제한 (테스트용)")
    parser.add_argument("--checkpoint_every",
                        type=int,
                        default=50,
                        help="체크포인트 저장 주기")
    args = parser.parse_args()

    evaluator = GEvalTyposEvaluator(model_name=args.model)

    if args.limit:
        print(f"  - 평가 제한: {args.limit}개 항목")

    evaluator.run_evaluation(
        data_file=args.input_file,
        output_file=args.output_file,
        max_workers=args.workers,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every
    )



if __name__ == "__main__":
    main()
