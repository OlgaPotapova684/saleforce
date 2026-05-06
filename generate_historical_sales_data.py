import csv
import random
from dataclasses import dataclass
from pathlib import Path

INDUSTRIES = ["Fintech", "SaaS", "Manufacturing", "Retail", "Logistics"]
LEAD_SOURCES = ["LinkedIn", "Cold Email", "Referral", "Website"]


@dataclass(frozen=True)
class CompanyNameParts:
    prefixes: tuple[str, ...]
    cores: tuple[str, ...]
    suffixes: tuple[str, ...]


NAME_PARTS = CompanyNameParts(
    prefixes=(
        "Northbridge",
        "Summit",
        "Silverline",
        "Crescent",
        "Blue Harbor",
        "Pioneer",
        "Redwood",
        "Atlas",
        "Oakridge",
        "Clearwater",
        "Evergreen",
        "Stonegate",
        "Apex",
        "Riverstone",
        "BrightPath",
        "Ironwood",
        "Horizon",
        "Keystone",
        "Beacon",
        "Golden Gate",
    ),
    cores=(
        "Payments",
        "Capital",
        "Analytics",
        "Logistics",
        "Manufacturing",
        "Retail",
        "Cloud",
        "Systems",
        "Solutions",
        "Supply",
        "Commerce",
        "Ledger",
        "Risk",
        "Ops",
        "Platform",
        "Automation",
        "Networks",
        "Advisory",
        "Dynamics",
        "Insights",
    ),
    suffixes=(
        "Inc.",
        "LLC",
        "Group",
        "Holdings",
        "Partners",
        "Labs",
        "Technologies",
        "Co.",
        "International",
        "Corp.",
    ),
)


def make_company_names(n: int, rng: random.Random) -> list[str]:
    names: set[str] = set()
    attempts = 0
    while len(names) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Failed to generate enough unique company names.")

        prefix = rng.choice(NAME_PARTS.prefixes)
        core = rng.choice(NAME_PARTS.cores)
        suffix = rng.choice(NAME_PARTS.suffixes)
        variant = rng.random()

        if variant < 0.5:
            name = f"{prefix} {core} {suffix}"
        elif variant < 0.8:
            name = f"{prefix} {core}"
        else:
            name = f"{core} {suffix}"

        if rng.random() < 0.3:
            name = name.replace(" ", "-")

        if rng.random() < 0.15:
            name = f"{name} {rng.choice(['AI', 'One', 'Next', '360', 'Prime'])}"

        names.add(name)

    return sorted(names)


def deal_probability(industry: str, emp_count: int, engagement_score: int) -> float:
    if industry == "Fintech" and emp_count > 200 and engagement_score > 60:
        return 0.85
    return 0.15


def main() -> None:
    rng = random.Random(42)

    out_path = Path(__file__).with_name("historical_sales_data.csv")
    n_rows = 500

    company_names = make_company_names(n_rows, rng)
    rng.shuffle(company_names)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "company_name",
                "industry",
                "emp_count",
                "revenue_mln",
                "engagement_score",
                "lead_source",
                "closed_deal",
            ],
        )
        writer.writeheader()

        for i in range(n_rows):
            industry = rng.choice(INDUSTRIES)
            emp_count = rng.randint(50, 5000)

            revenue_mln = rng.uniform(5, 500)
            revenue_mln = round(revenue_mln, 2)

            engagement_score = rng.randint(1, 100)
            lead_source = rng.choice(LEAD_SOURCES)

            p = deal_probability(industry, emp_count, engagement_score)
            closed_deal = 1 if rng.random() < p else 0

            writer.writerow(
                {
                    "company_name": company_names[i],
                    "industry": industry,
                    "emp_count": emp_count,
                    "revenue_mln": revenue_mln,
                    "engagement_score": engagement_score,
                    "lead_source": lead_source,
                    "closed_deal": closed_deal,
                }
            )

    print(f"Wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()

