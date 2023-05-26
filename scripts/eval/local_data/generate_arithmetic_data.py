import random
import json

def format_sample(a, b, c, operator="+"):
    # TODO: add aliases here so that the model gets points even if it is "close"
    return {
        "context": f"{a}+{b}=",
        "answer": str(c),
        "aliases": [str(c)]
    }


# generates a jsonl file of "a+b=c" samples where a, b are integers with num_digits digits
def make_arithmetic_dataset(out_filename, num_samples=1000, num_digits=3, random_subset=False):
    with open(out_filename, "w", encoding='utf8') as f:
        if random_subset:
            # then just pick num_samples randomly
            for idx in range(num_samples):
                # TODO: handle duplicates
                max_val = 10 ** num_digits
                a = random.randint(0, max_val)
                b = random.randint(0, max_val)
                c = a + b
                row = format_sample(a, b, c=c, operator="+")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            # consider all possible addition samples of num_digits
            for a in range(10**num_digits):
                for b in range(10**num_digits):
                    row = format_sample(a, b, c=a+b, operator="+")
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    num_digits = 3
    random_subset = False
    out_filename = f"{num_digits}_digits_addition.jsonl"
    if not random_subset:
        num_samples = 10 ** (2 * num_digits)
    print(f"Generating addition dataset with with {num_samples} samples of up to {num_digits} digits")
    make_arithmetic_dataset(out_filename, num_samples=num_samples, num_digits=num_digits, random_subset=random_subset)
