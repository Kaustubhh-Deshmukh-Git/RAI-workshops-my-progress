import json
from azure.ai.evaluation import F1ScoreEvaluator, RougeScoreEvaluator, RougeType, BleuScoreEvaluator, MeteorScoreEvaluator, GleuScoreEvaluator

# Sample data from the Contoso dining chair example
response = (
    "Introducing our timeless wooden dining chair, designed for both comfort and durability. "
    "Crafted with a solid wood seat and sturdy four-legged base, this chair offers reliable support for up to 250 lbs. "
    "The smooth brown finish adds a touch of rustic elegance, while the ergonomically shaped backrest ensures a comfortable dining experience. "
    "Measuring 18\" wide, 20\" deep, and 35\" tall, it's the perfect blend of form and function, making it a versatile addition to any dining space. "
    "Elevate your home with this beautifully simple yet sophisticated seating option."
)
ground_truth = (
    "The dining chair is brown and wooden with four legs and a backrest. "
    "The dimensions are 18\" wide, 20\" deep, 35\" tall. "
    "The dining chair has a weight capacity of 250 lbs."
)

print("--- Running Offline/Algorithmic Evaluators ---")

# 1. F1 Score
try:
    f1_eval = F1ScoreEvaluator()
    f1_score = f1_eval(response=response, ground_truth=ground_truth)
    print(f"F1 Score: {f1_score}")
except Exception as e:
    print(f"F1 Score Error: {e}")

# 2. ROUGE-1
try:
    rouge_eval = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_1)
    rouge_score = rouge_eval(response=response, ground_truth=ground_truth)
    print(f"ROUGE-1 Score: {rouge_score}")
except Exception as e:
    print(f"ROUGE Score Error: {e}")

# 3. BLEU
try:
    bleu_eval = BleuScoreEvaluator()
    bleu_score = bleu_eval(response=response, ground_truth=ground_truth)
    print(f"BLEU Score: {bleu_score}")
except Exception as e:
    print(f"BLEU Score Error: {e}")

# 4. METEOR
try:
    meteor_eval = MeteorScoreEvaluator(alpha=0.9, beta=3.0, gamma=0.5)
    meteor_score = meteor_eval(response=response, ground_truth=ground_truth)
    print(f"METEOR Score: {meteor_score}")
except Exception as e:
    print(f"METEOR Score Error: {e}")

# 5. GLEU
try:
    gleu_eval = GleuScoreEvaluator()
    gleu_score = gleu_eval(response=response, ground_truth=ground_truth)
    print(f"GLEU Score: {gleu_score}")
except Exception as e:
    print(f"GLEU Score Error: {e}")
