#!/usr/bin/env python3
"""
Простой тест для проверки логики расчёта стоимости с разными коэффициентами.
"""

def calculate_cost(price_info, prompt_tokens, completion_tokens):
    """Имитирует логику расчёта стоимости из responses_from_llm_chunk."""
    if isinstance(price_info, dict):
        # Разные цены для входных и выходных токенов
        input_coef = price_info.get("input", 0)
        output_coef = price_info.get("output", 0)
        expected_cost = (input_coef * prompt_tokens +
                        output_coef * completion_tokens)
    else:
        # Единая цена
        total_tokens = prompt_tokens + completion_tokens
        expected_cost = price_info * total_tokens
    return expected_cost


# Тест 1: openaivlm с разделением цен
print("Test 1: openaivlm with separated prices")
openaivlm_price = {
    "input": 0.00000175,
    "output": 0.000014
}
prompt_tokens = 100
completion_tokens = 50
cost = calculate_cost(openaivlm_price, prompt_tokens, completion_tokens)
expected = 0.00000175 * 100 + 0.000014 * 50
print(f"  Input tokens: {prompt_tokens}, Output tokens: {completion_tokens}")
print(f"  Cost: ${cost:.10f}")
print(f"  Expected: ${expected:.10f}")
assert cost == expected, f"Cost mismatch: {cost} != {expected}"
print("  ✓ PASSED\n")

# Тест 2: deepseek с разделением цен
print("Test 2: deepseek with separated prices")
deepseek_price = {
    "input": 0.00000028,
    "output": 0.00000042
}
prompt_tokens = 200
completion_tokens = 75
cost = calculate_cost(deepseek_price, prompt_tokens, completion_tokens)
expected = 0.00000028 * 200 + 0.00000042 * 75
print(f"  Input tokens: {prompt_tokens}, Output tokens: {completion_tokens}")
print(f"  Cost: ${cost:.10f}")
print(f"  Expected: ${expected:.10f}")
assert cost == expected, f"Cost mismatch: {cost} != {expected}"
print("  ✓ PASSED\n")

# Тест 3: yandexai с единой ценой (обратная совместимость)
print("Test 3: yandexai with single price (backward compatibility)")
yandexai_price = 1.666666e-06
prompt_tokens = 150
completion_tokens = 100
cost = calculate_cost(yandexai_price, prompt_tokens, completion_tokens)
expected = yandexai_price * (prompt_tokens + completion_tokens)
print(f"  Input tokens: {prompt_tokens}, Output tokens: {completion_tokens}")
print(f"  Cost: ${cost:.10f}")
print(f"  Expected: ${expected:.10f}")
assert cost == expected, f"Cost mismatch: {cost} != {expected}"
print("  ✓ PASSED\n")

# Тест 4: Старая структура openaivlm цена была единой (было бы без разделения)
print("Test 4: Old openaivlm single price (should cost more for same tokens)")
old_openaivlm_price = 0.000014  # была только цена выходных токенов
prompt_tokens = 100
completion_tokens = 50
old_cost = calculate_cost(old_openaivlm_price, prompt_tokens, completion_tokens)
print(f"  Old single price approach: ${old_cost:.10f}")

# Новый подход с разделением
new_openaivlm_price = {
    "input": 0.00000175,
    "output": 0.000014
}
new_cost = calculate_cost(new_openaivlm_price, prompt_tokens, completion_tokens)
print(f"  New separated prices approach: ${new_cost:.10f}")
print(f"  Difference: ${new_cost - old_cost:.10f}")
print("  (Input tokens are now much cheaper)\n")

print("All tests PASSED! ✓")
