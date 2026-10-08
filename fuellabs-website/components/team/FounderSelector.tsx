"use client";

import Image from "next/image";
import { useState } from "react";
import { founders } from "@/content/site";
import styles from "./FounderSelector.module.css";

export function FounderSelector() {
  const [active, setActive] = useState<(typeof founders)[number]["id"]>("prajhan");
  return <div className={styles.content}><section className={styles.portraits} aria-label="fuellabs. co-founders">{founders.map((founder) => <figure className={styles.portrait} key={founder.id}><Image src={`/portraits/${founder.id}${active === founder.id ? "-active" : ""}.png`} width={256} height={254} alt={founder.alt} priority /></figure>)}</section><section className={styles.selector} aria-label="Fuel Labs co-founders"><p>Co-founders</p>{founders.map((founder) => <a className={styles.profileLink} data-active={active === founder.id} href={founder.xUrl} key={founder.id} target="_blank" rel="noreferrer" onMouseEnter={() => setActive(founder.id)} onFocus={() => setActive(founder.id)} onClick={() => setActive(founder.id)}><span className={styles.founderName}>{founder.name}</span></a>)}</section></div>;
}
