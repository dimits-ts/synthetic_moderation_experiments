from pathlib import Path
from bs4 import BeautifulSoup

from syndisco.actors import Persona



def parse_profile_table(html: str) -> list[Persona]:
    """Parse the given HTML table into a list of Persona objects."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tbody tr")

    personas = []

    for row in rows:
        cols = row.select("td")
        if len(cols) < 4:
            continue

        # --- ID ---
        pid = cols[0].get_text(strip=True)

        # --- Demographics ---
        demo_divs = cols[1].select("div.text-sm > div")
        age, sex = None, None
        demographic_group = None
        if len(demo_divs) >= 3:
            # Example: "25-34 Male" or "Under 18 Female"
            age_sex = demo_divs[0].get_text(strip=True).split()
            if len(age_sex) >= 2:
                age_range = age_sex[0]
                if age_range.isdigit():
                    age = int(age_range)
                else:
                    # Use the midpoint if age is a range like "25-34"
                    import re

                    m = re.match(r"(\d+)-(\d+)", age_range)
                    if m:
                        age = (int(m.group(1)) + int(m.group(2))) // 2
                sex = age_sex[-1]
            demographic_group = demo_divs[1].get_text(strip=True)
            sexual_orientation = demo_divs[2].get_text(strip=True)
        else:
            sexual_orientation = ""

        # --- Background ---
        bg_divs = cols[2].select("div.text-sm > div")
        if len(bg_divs) >= 3:
            education_level = bg_divs[0].get_text(strip=True)
            current_employment = bg_divs[1].get_text(strip=True)
        else:
            education_level = current_employment = ""

        # --- Personality ---
        personality_desc = cols[3].select_one(".text-foreground\\/90")
        description = (
            personality_desc.get_text(strip=True) if personality_desc else ""
        )
        tags = [t.get_text(strip=True) for t in cols[3].select("span.rounded")]

        # Create Persona object
        persona = Persona(
            username=pid,
            age=age if age is not None else -1,
            sex=sex or "",
            sexual_orientation=sexual_orientation,
            demographic_group=demographic_group or "",
            current_employment=current_employment,
            education_level=education_level,
            special_instructions=description,
            personality_characteristics=tags,
        )

        personas.append(persona)

    return personas


def get_personas_from_html(html_dir: Path) -> list[Persona]:
    personas = []
    for file in html_dir.iterdir():
        html = file.read_text()
        personas += parse_profile_table(html)
    return personas
