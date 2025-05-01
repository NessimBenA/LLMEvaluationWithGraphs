import json
import os
import re
import argparse
import sys

def sanitize_json_content(content: str) -> tuple[str | None, str | None]:
    """
    Applies cleaning logic (fences, bracket extraction) and validates JSON.
    Returns (cleaned_json_string, error_message)
    """
    original_content = content
    processed_content = content.strip()
    cleaning_step = "original"

    # Step 1: Clean fences
    if processed_content.startswith("```json") and processed_content.endswith("```"):
        processed_content = processed_content[7:-3].strip()
        cleaning_step = "cleaned ```json fences"
    elif processed_content.startswith("```") and processed_content.endswith("```"):
        processed_content = processed_content[3:-3].strip()
        cleaning_step = "cleaned ``` fences"

    if not processed_content:
        return None, f"Content empty after {cleaning_step}."

    # Step 2: Extract bracket content
    first_brace = processed_content.find('{')
    last_brace = processed_content.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_str = processed_content[first_brace : last_brace + 1]
        cleaning_step += ", then extracted bracket content" if cleaning_step != "original" else "extracted bracket content"
    else:
        # If braces not found after potential fence cleaning, assume it might be invalid
        # Or maybe it was *just* the JSON without extra text/fences
        potential_json_str = processed_content
        if '{' not in potential_json_str and '}' not in potential_json_str:
             return None, f"No JSON object braces found after {cleaning_step}."
        # If braces ARE present but not at start/end, json.loads will fail later, which is ok

    # Step 3: Validate by parsing and re-dumping
    try:
        if not potential_json_str.strip():
             return None, f"Extracted content is empty after {cleaning_step}."
        parsed_data = json.loads(potential_json_str)
        # Re-dump with indentation for pretty-printing and consistency
        cleaned_json_string = json.dumps(parsed_data, indent=2)
        return cleaned_json_string, None # Success
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError after {cleaning_step}: {e}. Preview: {potential_json_str[:100]}..."
    except Exception as e:
        return None, f"Unexpected validation error after {cleaning_step}: {e}"


def process_directory(directory: str, dry_run: bool = False):
    """Iterates through .json files, sanitizes, and overwrites."""
    print(f"--- Starting JSON Sanitization {'(Dry Run)' if dry_run else ''} ---")
    print(f"Target Directory: {directory}")

    cleaned_count = 0
    error_count = 0
    skipped_count = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".json"):
                filepath = os.path.join(root, filename)
                print(f"\nProcessing: {filepath}")

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        original_content = f.read()

                    if not original_content.strip():
                        print("  Skipped: File is empty.")
                        skipped_count += 1
                        continue

                    cleaned_json, error = sanitize_json_content(original_content)

                    if error:
                        print(f"  Error: {error}")
                        # Optionally log original preview for context
                        # print(f"    Original Preview: {original_content[:150]}...")
                        error_count += 1
                        continue

                    if cleaned_json is None: # Should not happen if error is None, but safety check
                         print(f"  Error: Sanitization returned None without error message.")
                         error_count += 1
                         continue

                    # Check if content actually changed (ignoring whitespace differences from indent)
                    # Parse original again (best effort) to compare structure, not just string
                    original_parsed = None
                    try:
                        # Try parsing the result of bracket extraction on original for comparison
                        # This handles cases where original had fences/text but was otherwise identical
                        _, original_error = sanitize_json_content(original_content)
                        if not original_error:
                           temp_cleaned_original, _ = sanitize_json_content(original_content)
                           if temp_cleaned_original:
                               original_parsed = json.loads(temp_cleaned_original) # Load the *cleaned* original
                    except:
                        pass # Ignore errors parsing original for comparison

                    cleaned_parsed = json.loads(cleaned_json) # We know this works

                    # Only write if structure differs or original wasn't valid/parsable cleanly
                    needs_write = (original_parsed != cleaned_parsed) or (original_parsed is None)

                    if needs_write:
                        if not dry_run:
                            try:
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(cleaned_json)
                                print(f"  Success: Cleaned and overwrote file.")
                                cleaned_count += 1
                            except IOError as e:
                                print(f"  Error: Failed to write cleaned file: {e}")
                                error_count += 1
                        else:
                            print(f"  Success: File needs cleaning (Dry Run - Not saving).")
                            cleaned_count += 1 # Count as needing cleaning
                    else:
                         print(f"  Skipped: File content already clean.")
                         skipped_count += 1


                except IOError as e:
                    print(f"  Error: Cannot read file: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"  Error: Unexpected failure processing file: {e}")
                    error_count += 1

    print("\n--- Sanitization Summary ---")
    print(f"Files Processed: {cleaned_count + error_count + skipped_count}")
    print(f"Files Cleaned/Overwritten: {cleaned_count if not dry_run else 0} ({cleaned_count} needed cleaning)")
    print(f"Files Skipped (Empty/Already Clean): {skipped_count}")
    print(f"Files with Errors: {error_count}")
    print("--- Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize JSON files in an output directory.")
    parser.add_argument("directory", help="Path to the directory containing JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    process_directory(args.directory, args.dry_run)