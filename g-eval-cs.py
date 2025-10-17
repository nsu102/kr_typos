"""
G-eval을 사용한 코드 스위칭 데이터 평가 시스템
5C2 방식으로 Case들끼리 비교하여 어떤 응답이 더 좋은지 평가
"""

import json
import os
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
from tqdm import tqdm
from itertools import combinations

# OpenRouter API 설정
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

@dataclass
class ComparisonResult:
    """비교 결과를 담는 데이터 클래스"""
    original_question: str  # original_ko
    original_en: str
    case_a: str
    case_b: str
    model: str
    question_a: str
    question_b: str
    response_a: str
    response_b: str
    score_a: float
    score_b: float
    metrics_a: Dict  # 개별 메트릭 점수
    metrics_b: Dict  # 개별 메트릭 점수


class GEvalCodeSwitchingEvaluator:
    """G-eval을 사용한 코드 스위칭 응답 평가기"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.g_eval_prompt = self._create_g_eval_prompt()

    def _create_g_eval_prompt(self) -> str:
        """G-EVAL exact prompt format (English)"""
        return """You are given one original question and two code-switching versions of the question, along with responses to each.
Your task is to rate each response according to four evaluation criteria.
Please read and understand these instructions carefully. Keep this document open while evaluating and refer to it as needed.

Evaluation Criteria:

Helpfulness (1-10) - The ability of the response to effectively meet the user's needs despite code-switching. Aligns with the quality of providing practical, actionable, and insightful information in that "the response should directly answer the question and provide useful information that the user can apply or understand, regardless of the mixed-language input."

Relevance (1-10) - How well the response stays on topic. Aligns with focus and coherence in that "the response should directly address the question without unnecessary digressions or irrelevant information, maintaining a clear connection to the question throughout."

Accuracy (1-10) - The factual correctness and reliability of the information provided. Aligns with truthfulness and precision in that "the response should not contain factual errors, misinformation, or misleading statements, and should present verifiable and trustworthy information."

Depth (1-10) - The level of detail and comprehensiveness in the response. Aligns with thoroughness and insight in that "the response should go beyond surface-level information to provide detailed explanations, meaningful context, relevant examples, and thoughtful analysis that demonstrates deep understanding of the topic."

Evaluation Steps:

1. Read the original question (in Korean and English) and identify the main topic and key points.
2. Read code-switching question A and note the code-switching pattern used.
3. Read response A and compare it to the original question. Check whether the response demonstrates understanding of the code-switching input, effectively meets the user's needs, maintains relevance to the topic, provides accurate information, and offers sufficient depth.
4. Assign a score from 1 to 10 for each evaluation criterion (Helpfulness, Relevance, Accuracy, Depth) for response A. 1 is the lowest, 10 is the highest.
5. Read code-switching question B and note the code-switching pattern used.
6. Read response B and compare it to the original question. Check whether the response demonstrates understanding of the code-switching input, effectively meets the user's needs, maintains relevance to the topic, provides accurate information, and offers sufficient depth.
7. Assign a score from 1 to 10 for each evaluation criterion (Helpfulness, Relevance, Accuracy, Depth) for response B. 1 is the lowest, 10 is the highest.

Example:

Original Question (Korean):
{original_ko}

Original Question (English):
{original_en}

Code-switching Question A ({case_a}):
{question_a}

Response A:
{response_a}

Code-switching Question B ({case_b}):
{question_b}

Response B:
{response_b}

Evaluation Form (scores only):

{{
    "helpfulness_a": <score 1-10>,
    "relevance_a": <score 1-10>,
    "accuracy_a": <score 1-10>,
    "depth_a": <score 1-10>,
    "helpfulness_b": <score 1-10>,
    "relevance_b": <score 1-10>,
    "accuracy_b": <score 1-10>,
    "depth_b": <score 1-10>
}}
"""

    def evaluate_pair(
        self,
        original_ko: str,
        original_en: str,
        case_a: str,
        question_a: str,
        model_a: str,
        response_a: str,
        case_b: str,
        question_b: str,
        model_b: str,
        response_b: str
    ) -> Dict:
        """두 응답을 G-eval로 비교 평가"""

        prompt = self.g_eval_prompt.format(
            original_ko=original_ko,
            original_en=original_en,
            case_a=case_a,
            question_a=question_a,
            model_a=model_a,
            response_a=response_a[:2000],
            case_b=case_b,
            question_b=question_b,
            model_b=model_b,
            response_b=response_b[:2000]
        )

        # Retry logic for rate limits
        max_retries = 5
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "당신은 언어 모델 응답의 품질을 평가하는 전문 평가자입니다. 반드시 유효한 JSON 객체로 응답해야 합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content.strip()
                result = json.loads(result_text)
                break

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⏳ Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Max retries reached for rate limit error")
                        raise
                else:
                    raise

        try:
            # 4개 메트릭의 평균으로 최종 점수 계산
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

            return {
                "score_a": score_a,
                "score_b": score_b,
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
                "metrics_a": {"helpfulness": 5, "relevance": 5, "accuracy": 5, "depth": 5},
                "metrics_b": {"helpfulness": 5, "relevance": 5, "accuracy": 5, "depth": 5}
            }

    def load_data(self, file_path: str) -> List[Dict]:
        """JSON 데이터 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_comparison_pairs(self, data: List[Dict]) -> List[Dict]:
        """비교 쌍 생성: 5C2 방식으로 Case들끼리 비교"""
        comparison_pairs = []
        CASES = ['Case1', 'Case2', 'Case3', 'Case4', 'Case5']
        MODELS = ['qwen_72b', 'qwen_7b']

        for item in data:
            question_id = item['id']
            original_ko = item['original_ko']
            original_en = item['original_en']

            # 각 모델에 대해
            for model in MODELS:
                # 5개 Case 중 2개를 선택 (5C2 = 10)
                for case_a, case_b in combinations(CASES, 2):
                    # Case A 데이터 확인
                    if case_a not in item:
                        continue
                    entry_a = item[case_a]
                    if 'responses' not in entry_a or model not in entry_a['responses']:
                        continue

                    # Case B 데이터 확인
                    if case_b not in item:
                        continue
                    entry_b = item[case_b]
                    if 'responses' not in entry_b or model not in entry_b['responses']:
                        continue

                    # 비교 쌍 생성
                    comparison_pairs.append({
                        'question_id': question_id,
                        'original_ko': original_ko,
                        'original_en': original_en,
                        'case_a': case_a,
                        'case_b': case_b,
                        'model': model,
                        'question_a': entry_a['text'],
                        'question_b': entry_b['text'],
                        'response_a': entry_a['responses'][model],
                        'response_b': entry_b['responses'][model]
                    })

        return comparison_pairs

    def process_single_comparison(self, pair: Dict) -> ComparisonResult:
        """단일 비교 쌍 처리"""
        eval_result = self.evaluate_pair(
            original_ko=pair['original_ko'],
            original_en=pair['original_en'],
            case_a=pair['case_a'],
            question_a=pair['question_a'],
            model_a=pair['model'],
            response_a=pair['response_a'],
            case_b=pair['case_b'],
            question_b=pair['question_b'],
            model_b=pair['model'],
            response_b=pair['response_b']
        )

        return ComparisonResult(
            original_question=pair['original_ko'],
            original_en=pair['original_en'],
            case_a=pair['case_a'],
            case_b=pair['case_b'],
            model=pair['model'],
            question_a=pair['question_a'],
            question_b=pair['question_b'],
            response_a=pair['response_a'],
            response_b=pair['response_b'],
            score_a=eval_result['score_a'],
            score_b=eval_result['score_b'],
            metrics_a=eval_result['metrics_a'],
            metrics_b=eval_result['metrics_b']
        )

    def run_evaluation(
        self,
        data_file: str,
        output_file: str = "data/outputs/cs/g_eval_results.json",
        max_workers: int = 3,
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

        # 최종 결과 저장
        print(f"💾 최종 결과 저장 중: {output_file}")
        self._save_results(results, output_file)

        # 체크포인트 파일 삭제
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️  체크포인트 파일 삭제: {checkpoint_file}")

        # 통계 출력
        self._print_statistics(results)

        return results

    def _save_checkpoint(self, results: List[ComparisonResult], checkpoint_file: str):
        """체크포인트 저장"""
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        results_dict = [
            {
                'original_question': r.original_question,
                'original_en': r.original_en,
                'case_a': r.case_a,
                'case_b': r.case_b,
                'model': r.model,
                'question_a': r.question_a,
                'question_b': r.question_b,
                'response_a': r.response_a,
                'response_b': r.response_b,
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
                'original_en': r.original_en,
                'case_a': r.case_a,
                'case_b': r.case_b,
                'model': r.model,
                'question_a': r.question_a,
                'question_b': r.question_b,
                'response_a': r.response_a,
                'response_b': r.response_b,
                'score_a': r.score_a,
                'score_b': r.score_b,
                'metrics_a': r.metrics_a,
                'metrics_b': r.metrics_b
            }
            for r in results
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    def _print_statistics(self, results: List[ComparisonResult]):
        """평가 통계 출력"""
        print("\n" + "="*70)
        print("📈 G-Eval 코드 스위칭 평가 결과 통계")
        print("="*70)

        total = len(results)

        # 모델별 통계
        for model in ['qwen_72b', 'qwen_7b']:
            filtered = [r for r in results if r.model == model]
            if not filtered:
                continue

            count = len(filtered)
            avg_score_a = sum(r.score_a for r in filtered) / count
            avg_score_b = sum(r.score_b for r in filtered) / count

            print(f"📊 {model} 통계 (n={count}):")
            print(f"  - Case A 평균: {avg_score_a:.3f}/10.0")
            print(f"  - Case B 평균: {avg_score_b:.3f}/10.0")

        # 전체 평균 점수
        avg_score_a = sum(r.score_a for r in results) / total
        avg_score_b = sum(r.score_b for r in results) / total

        print(f"⭐ 전체 평균 점수 (1-10 척도):")
        print(f"  - Case A (첫번째 Case): {avg_score_a:.3f}/10.0")
        print(f"  - Case B (두번째 Case): {avg_score_b:.3f}/10.0")

        print(f"📊 평가 기준별 평균 점수:")

        avg_helpfulness_a = sum(r.metrics_a['helpfulness'] for r in results) / total
        avg_relevance_a = sum(r.metrics_a['relevance'] for r in results) / total
        avg_accuracy_a = sum(r.metrics_a['accuracy'] for r in results) / total
        avg_depth_a = sum(r.metrics_a['depth'] for r in results) / total

        avg_helpfulness_b = sum(r.metrics_b['helpfulness'] for r in results) / total
        avg_relevance_b = sum(r.metrics_b['relevance'] for r in results) / total
        avg_accuracy_b = sum(r.metrics_b['accuracy'] for r in results) / total
        avg_depth_b = sum(r.metrics_b['depth'] for r in results) / total

        print("  Case A (첫번째 Case):")
        print(f"    - 유용성 (Helpfulness): {avg_helpfulness_a:.2f}/10.0")
        print(f"    - 관련성 (Relevance): {avg_relevance_a:.2f}/10.0")
        print(f"    - 정확성 (Accuracy): {avg_accuracy_a:.2f}/10.0")
        print(f"    - 깊이 (Depth): {avg_depth_a:.2f}/10.0")
        print("  Case B (두번째 Case):")
        print(f"    - 유용성 (Helpfulness): {avg_helpfulness_b:.2f}/10.0")
        print(f"    - 관련성 (Relevance): {avg_relevance_b:.2f}/10.0")
        print(f"    - 정확성 (Accuracy): {avg_accuracy_b:.2f}/10.0")
        print(f"    - 깊이 (Depth): {avg_depth_b:.2f}/10.0")

        print(f"\n📊 종합 분석:")
        print(f"  - 점수 차이: {abs(avg_score_a - avg_score_b):.3f}")
        print("="*70)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="G-Eval을 사용한 코드 스위칭 응답 평가")
    parser.add_argument("--input_file",
                        default="data/outputs/cs/code_switched_response.json.ckpt.json",
                        help="입력 데이터 파일")
    parser.add_argument("--output_file",
                        default="data/outputs/cs/g_eval_results.json",
                        help="출력 결과 파일")
    parser.add_argument("--model",
                        default="openai/gpt-4o-mini",
                        help="평가에 사용할 모델")
    parser.add_argument("--workers",
                        type=int,
                        default=3,
                        help="멀티스레딩 워커 수")
    parser.add_argument("--limit",
                        type=int,
                        default=None,
                        help="평가할 항목 수 제한")
    parser.add_argument("--checkpoint_every",
                        type=int,
                        default=50,
                        help="체크포인트 저장 주기")
    args = parser.parse_args()

    # 평가기 초기화
    evaluator = GEvalCodeSwitchingEvaluator(model_name=args.model)

    print(f"📊 평가 설정:")
    print(f"  - 입력 파일: {args.input_file}")
    print(f"  - 출력 파일: {args.output_file}")
    print(f"  - 모델: {args.model}")
    print(f"  - 워커 수: {args.workers}")
    print(f"  - 체크포인트 주기: {args.checkpoint_every}개마다")
    if args.limit:
        print(f"  - 평가 제한: {args.limit}개 항목")

    # 평가 실행
    results = evaluator.run_evaluation(
        data_file=args.input_file,
        output_file=args.output_file,
        max_workers=args.workers,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every
    )

    print(f"\n✅ 평가 완료! 결과는 {args.output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
