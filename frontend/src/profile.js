// Count Unicode code points, matching Python string length (not UTF-16 units).
export const countBioCharacters = bio => Array.from(bio).length;
export function profilePayload(form) {
  const {bio, ...profile} = form;
  return {
    ...profile,
    ...Object.fromEntries(['followers','following','posts','account_age_days','username_changed'].map(key => [key, Number(profile[key])])),
    bio_length: countBioCharacters(bio),
  };
}
