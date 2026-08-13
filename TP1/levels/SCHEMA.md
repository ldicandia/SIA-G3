# Grid World Level File Format (SCHEMA)

A Grid World level file is a UTF-8 JSON file residing under `TP1/levels/` that declares a board's dimensions, obstacles, starting car positions, and target flag positions.

## Fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | string | Yes | Human-readable name for the level. |
| `rows` | integer | Yes | Number of rows in the grid (1 to 100). |
| `cols` | integer | Yes | Number of columns in the grid (1 to 100). |
| `obstacles` | list of coordinates | Yes | List of black obstacle grid cells. |
| `cars` | list of objects | Yes | List of starting car positions. Each object must have `number` (positive int) and `at` (coordinate). |
| `flags` | list of objects | Yes | List of destination flag positions. Each object must have `number` (positive int) and `at` (coordinate). |

## Coordinate Convention

Coordinates are declared as two-element arrays: `[row, col]`

- **`row 0`** is the top row of the grid.
- **`col 0`** is the left-most column of the grid.
- **`[row, col]`** specifies grid row first, then column. It is **not** `[x, y]`.

## Cars and Flags

Cars and flags are declared as separate lists:
- Each car entry has a 1-based `number` and starting `at` coordinate.
- Each flag entry has a matching 1-based `number` and destination `at` coordinate.
- Car number `n` must be moved to flag number `n` to complete the level.

## Limits

- **Grid Dimensions**: Grid sides must be integers between 1 and 100 inclusive (`MAX_DIMENSION = 100`).
- **Rendering**: The pygame renderer enforces a 24-pixel minimum cell size and clips grids larger than the window area.

## Worked Example

Below is the complete content of `TP1/levels/01-warmup.json`:

```json
{
  "name": "Warmup",
  "rows": 5,
  "cols": 5,
  "obstacles": [
    [2, 1],
    [2, 3]
  ],
  "cars": [
    {
      "number": 1,
      "at": [0, 0]
    },
    {
      "number": 2,
      "at": [0, 4]
    }
  ],
  "flags": [
    {
      "number": 1,
      "at": [4, 4]
    },
    {
      "number": 2,
      "at": [4, 0]
    }
  ]
}
```

## Validation errors

When a level file fails loading or integrity validation, a `LevelError` is raised containing a specific `LevelProblem` code and detailed message naming the offending entity, value, or cell.

| Problem (`LevelProblem`) | Trigger Condition | Example Detail Message |
|--------------------------|-------------------|------------------------|
| `UNREADABLE_FILE` | File does not exist, path is invalid, or permission denied. | `cannot read level file: [Errno 2] No such file or directory` |
| `MALFORMED_JSON` | File content is not valid JSON syntax. | `line 3 column 1 (char 25)` |
| `NOT_AN_OBJECT` | Top-level JSON data is not an object/dictionary. | `level data must be a JSON object, got list` |
| `MISSING_KEY` | Required key is omitted from object or entry. | `missing required key 'rows'` |
| `WRONG_TYPE` | Key value has wrong type (e.g. non-string name, non-int rows/cols/number). | `key 'rows' must be an integer, got 5.5` |
| `BAD_DIMENSION` | Grid dimension outside range [1, 100]. | `key 'rows' must be between 1 and 100, got 0` |
| `BAD_COORDINATE` | Coordinate is not a 2-element list of integers. | `obstacle[0] coordinate must be a two-element [row, col] array, got [1, 2, 3]` |
| `OUT_OF_BOUNDS` | Coordinate lies outside the declared grid bounds. | `obstacle[0] coordinate [9, 0] is out of bounds for 3x3 grid` |
| `NO_CARS` | Level declares zero cars. | `level must declare at least one car` |
| `DUPLICATE_NUMBER` | Two cars or two flags share the same number. | `duplicate car number 1` |
| `NON_CONTIGUOUS_NUMBERING` | Car or flag numbers are not contiguous starting from 1. | `car numbers must be 1 to 2 with no gaps, broken by number 2 (found [1, 3])` |
| `UNPAIRED_NUMBER` | Car has no matching flag, or flag has no matching car. | `car 2 has no matching flag 2` |
| `ON_OBSTACLE` | Car or flag is placed on an obstacle cell. | `car 1 at [1, 1] is on an obstacle cell` |
| `CELL_CONFLICT` | Two cars, two flags, or a car and a foreign flag share a cell. | `car 1 and car 2 share cell [0, 0]` |
| `STARTS_ON_OWN_FLAG` | Car starts on its own destination flag. | `car 1 starts on its own flag at [2, 2]; level must not begin already solved` |

---
*Note: Phase 4 delivery README links to this document for the level specification.*
